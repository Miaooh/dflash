import argparse
import gc
import io
import time
import torch
import grpc
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, DynamicCache

import dflash_service_pb2
import dflash_service_pb2_grpc

from dflash.model import sample

TARGET_PATH = "/home/xzh/models/Qwen3-4B"
DRAFT_PATH = "/home/xzh/models/z-lab/Qwen3-4B-DFlash-b16"
DEVICE = "cuda:0"
CLOUD_ADDR = "192.168.100.2:50051"
GRPC_OPTIONS = [
    ("grpc.max_send_message_length", 256 * 1024 * 1024),
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),
]


def tensor_to_proto(tensor):
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return dflash_service_pb2.TensorProto(
        data=buffer.getvalue(),
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
    )


def proto_to_tensor(proto, device):
    buffer = io.BytesIO(proto.data)
    tensor = torch.load(buffer, map_location="cpu")
    return tensor.to(device)


def build_long_prompt(tokenizer, target_length):
    """Build a prompt of roughly target_length tokens by repeating a base paragraph."""
    base = (
        "DFlash is a lightweight block diffusion draft model designed for speculative decoding. "
        "It predicts multiple future tokens in parallel as a block, and a larger target model verifies them. "
    )
    base_ids = tokenizer.encode(base, add_special_tokens=False)
    repeats = max(1, target_length // len(base_ids))
    long_text = (base * repeats).strip()
    messages = [{"role": "user", "content": long_text + "\n\nPlease summarize the above text."}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return input_ids.to(DEVICE)


def dflash_generate_edge_cloud(
    draft,
    target_embed,
    target_lm_head,
    stub,
    input_ids,
    max_new_tokens,
    stop_token_ids,
    temperature,
    block_size=None,
    mask_token_id=None,
):
    from dflash.model import sample as dflash_sample

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = draft.block_size if block_size is None else block_size
    mask_token_id = draft.mask_token_id if mask_token_id is None else mask_token_id

    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=DEVICE,
    )
    position_ids = torch.arange(output_ids.shape[1], device=DEVICE).unsqueeze(0)
    past_key_values_draft = DynamicCache()

    t0 = time.perf_counter()
    prefill_response = stub.Prefill(
        dflash_service_pb2.PrefillRequest(
            input_ids=tensor_to_proto(input_ids),
            temperature=temperature,
        )
    )
    prefill_time = time.perf_counter() - t0
    target_hidden = proto_to_tensor(prefill_response.hidden_states, DEVICE)

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = prefill_response.first_token_id

    stats = {
        "rounds": 0,
        "prefill_time": prefill_time,
        "total_hidden_states_bytes": len(prefill_response.hidden_states.data),
        "total_token_candidates_bytes": 0,
        "total_verify_bytes": 0,
        "total_rpc_time": prefill_time,
        "total_draft_time": 0.0,
        "acceptance_lengths": [],
    }

    start = num_input_tokens

    while start < max_length:
        stats["rounds"] += 1

        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]

        if block_size > 1:
            t0 = time.perf_counter()
            noise_embedding = target_embed(block_output_ids)
            draft_output = draft(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )
            past_key_values_draft.crop(start)
            draft_logits = target_lm_head(draft_output[:, 1 - block_size :, :])
            block_output_ids[:, 1:] = dflash_sample(draft_logits, temperature)
            draft_time = time.perf_counter() - t0
            stats["total_draft_time"] += draft_time
            stats["total_token_candidates_bytes"] += len(tensor_to_proto(block_output_ids).data)

        t0 = time.perf_counter()
        verify_response = stub.Verify(
            dflash_service_pb2.VerifyRequest(
                candidate_ids=tensor_to_proto(block_output_ids),
                position_ids=tensor_to_proto(block_position_ids),
                crop_to=start,
                temperature=temperature,
            )
        )
        verify_rpc_time = time.perf_counter() - t0
        stats["total_verify_bytes"] += len(verify_response.hidden_states.data)
        stats["total_rpc_time"] += verify_rpc_time
        stats["total_hidden_states_bytes"] += len(verify_response.hidden_states.data)

        target_hidden = proto_to_tensor(verify_response.hidden_states, DEVICE)
        acceptance_length = verify_response.acceptance_length
        corrected_token = verify_response.corrected_token

        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, :acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = corrected_token
        start += acceptance_length + 1
        stats["acceptance_lengths"].append(acceptance_length + 1)

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :min(start + 1, max_length)]
    return output_ids, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-length", type=int, default=26, help="Target prompt length in tokens")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH)

    print("Loading draft model...")
    draft = AutoModel.from_pretrained(
        DRAFT_PATH,
        trust_remote_code=True,
        dtype="auto",
        device_map=DEVICE,
    ).eval()

    print("Loading target embedding and lm_head on edge...")
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH,
        dtype="auto",
        device_map=DEVICE,
    ).eval()
    target_embed = target_model.model.embed_tokens
    target_lm_head = target_model.lm_head

    channel = grpc.insecure_channel(CLOUD_ADDR, options=GRPC_OPTIONS)
    stub = dflash_service_pb2_grpc.DFlashCloudStub(channel)

    if args.prompt_length <= 26:
        messages = [{"role": "user", "content": "How many positive whole-number divisors does 196 have?"}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(DEVICE)
    else:
        input_ids = build_long_prompt(tokenizer, args.prompt_length)

    actual_prompt_len = input_ids.shape[1]
    print(f"Actual prompt length: {actual_prompt_len} tokens")
    print(f"Target max new tokens: {args.max_new_tokens}")

    torch.cuda.synchronize()
    start = time.perf_counter()
    output, stats = dflash_generate_edge_cloud(
        draft=draft,
        target_embed=target_embed,
        target_lm_head=target_lm_head,
        stub=stub,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=0.0,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    num_output = output.shape[1] - actual_prompt_len

    print(f"\nGenerated {num_output} tokens in {elapsed:.2f} s ({num_output / elapsed:.2f} tok/s)")
    print(f"Rounds: {stats['rounds']}")
    print(f"Avg acceptance length: {sum(stats['acceptance_lengths']) / len(stats['acceptance_lengths']):.2f}")
    print(f"Prefill RPC time: {stats['prefill_time']:.3f} s")
    print(f"Total RPC time: {stats['total_rpc_time']:.2f} s ({100 * stats['total_rpc_time'] / elapsed:.1f}%)")
    print(f"Total draft time: {stats['total_draft_time']:.2f} s")
    print(f"Total hidden states bytes: {stats['total_hidden_states_bytes'] / 1024**2:.2f} MB")
    print(f"Total verify response bytes: {stats['total_verify_bytes'] / 1024**2:.2f} MB")

    del draft, target_model, output
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
