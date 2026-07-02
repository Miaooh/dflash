import gc
import io
import time
import argparse
import torch
import grpc
from transformers import AutoModel, AutoTokenizer, DynamicCache

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
    """Serialize a CPU tensor to protobuf bytes."""
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return dflash_service_pb2.TensorProto(
        data=buffer.getvalue(),
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
    )


def proto_to_tensor(proto, device):
    """Deserialize protobuf bytes back to a tensor on the target device."""
    buffer = io.BytesIO(proto.data)
    tensor = torch.load(buffer, map_location="cpu")
    return tensor.to(device)


def rpc_call_time(start):
    return time.perf_counter() - start


def dflash_generate_edge_cloud(
    draft,
    tokenizer,
    stub,
    input_ids,
    max_new_tokens,
    stop_token_ids,
    temperature,
    block_size=None,
    mask_token_id=None,
):
    """Edge-side DFlash generation using cloud target via gRPC (Method 2).

    In this mode the edge does NOT hold target.embed_tokens or target.lm_head.
    Every draft step therefore requires two extra RPCs:
        1. GetEmbedding(block_output_ids) -> noise_embedding
        2. GetLogits(draft_output) -> draft_logits
    """
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = draft.block_size if block_size is None else block_size
    mask_token_id = draft.mask_token_id if mask_token_id is None else mask_token_id

    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=DEVICE,
    )
    position_ids = torch.arange(output_ids.shape[1], device=DEVICE).unsqueeze(0)
    past_key_values_draft = DynamicCache()

    # --- Prefill on cloud ---
    t0 = time.perf_counter()
    prefill_response = stub.Prefill(
        dflash_service_pb2.PrefillRequest(input_ids=tensor_to_proto(input_ids))
    )
    prefill_time = time.perf_counter() - t0
    target_hidden = proto_to_tensor(prefill_response.hidden_states, DEVICE)
    first_token_logits = proto_to_tensor(prefill_response.first_token_logits, DEVICE)

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(first_token_logits, temperature)

    stats = {
        "rounds": 0,
        "prefill_time": prefill_time,
        "total_hidden_states_bytes": len(prefill_response.hidden_states.data),
        "total_token_candidates_bytes": len(prefill_response.first_token_logits.data),
        "total_embedding_bytes": 0,
        "total_logits_bytes": 0,
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
            # RPC 1: cloud -> edge, get noise embedding
            t0 = time.perf_counter()
            emb_response = stub.GetEmbedding(
                dflash_service_pb2.EmbeddingRequest(token_ids=tensor_to_proto(block_output_ids))
            )
            emb_rpc_time = time.perf_counter() - t0
            stats["total_embedding_bytes"] += len(emb_response.embeddings.data)
            stats["total_rpc_time"] += emb_rpc_time
            noise_embedding = proto_to_tensor(emb_response.embeddings, DEVICE)

            # Edge: run draft
            t0 = time.perf_counter()
            draft_output = draft(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )
            past_key_values_draft.crop(start)
            draft_hidden_for_logits = draft_output[:, 1 - block_size :, :]

            # RPC 2: edge -> cloud, get draft logits from lm_head
            t1 = time.perf_counter()
            logits_response = stub.GetLogits(
                dflash_service_pb2.LogitsRequest(hidden_states=tensor_to_proto(draft_hidden_for_logits))
            )
            logits_rpc_time = time.perf_counter() - t1
            stats["total_logits_bytes"] += len(logits_response.logits.data)
            stats["total_rpc_time"] += logits_rpc_time
            draft_logits = proto_to_tensor(logits_response.logits, DEVICE)

            block_output_ids[:, 1:] = sample(draft_logits, temperature)
            draft_time = time.perf_counter() - t0
            stats["total_draft_time"] += draft_time

            stats["total_token_candidates_bytes"] += (
                len(tensor_to_proto(block_output_ids).data)
            )

        # RPC 3: cloud verify candidates and return hidden states for next round
        t0 = time.perf_counter()
        verify_response = stub.Verify(
            dflash_service_pb2.VerifyRequest(
                candidate_ids=tensor_to_proto(block_output_ids),
                position_ids=tensor_to_proto(block_position_ids),
                crop_to=start,
            )
        )
        verify_rpc_time = time.perf_counter() - t0
        stats["total_verify_bytes"] += (
            len(verify_response.logits.data) + len(verify_response.hidden_states.data)
        )
        stats["total_rpc_time"] += verify_rpc_time
        stats["total_hidden_states_bytes"] += len(verify_response.hidden_states.data)

        posterior_logits = proto_to_tensor(verify_response.logits, DEVICE)
        target_hidden = proto_to_tensor(verify_response.hidden_states, DEVICE)

        posterior = sample(posterior_logits, temperature)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        target_hidden = target_hidden[:, : acceptance_length + 1, :]
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1
        stats["acceptance_lengths"].append(acceptance_length + 1)

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :min(start + 1, max_length)]
    if stop_token_ids is not None:
        stop_token_ids_t = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_t).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    return output_ids, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH)

    print("Loading draft model...")
    draft = AutoModel.from_pretrained(
        DRAFT_PATH,
        trust_remote_code=True,
        dtype="auto",
        device_map=DEVICE,
    ).eval()

    print(f"Connecting to cloud target at {CLOUD_ADDR}...")
    channel = grpc.insecure_channel(CLOUD_ADDR, options=GRPC_OPTIONS)
    stub = dflash_service_pb2_grpc.DFlashCloudStub(channel)

    messages = [
        {"role": "user", "content": "How many positive whole-number divisors does 196 have?"}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(DEVICE)

    print(f"Input length: {input_ids.shape[1]} tokens")

    torch.cuda.synchronize()
    start = time.perf_counter()
    output, stats = dflash_generate_edge_cloud(
        draft=draft,
        tokenizer=tokenizer,
        stub=stub,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=0.0,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    num_input = input_ids.shape[1]
    num_output = output.shape[1] - num_input

    print(f"\nGenerated {num_output} tokens in {elapsed:.2f} s ({num_output / elapsed:.2f} tok/s)")
    print(f"Rounds: {stats['rounds']}")
    print(f"Avg acceptance length: {sum(stats['acceptance_lengths']) / len(stats['acceptance_lengths']):.2f}")
    print(f"Prefill RPC time: {stats['prefill_time']:.3f} s")
    print(f"Total RPC time: {stats['total_rpc_time']:.2f} s ({100 * stats['total_rpc_time'] / elapsed:.1f}%)")
    print(f"Total draft time: {stats['total_draft_time']:.2f} s")
    print(f"Total hidden states bytes: {stats['total_hidden_states_bytes'] / 1024**2:.2f} MB")
    print(f"Total token candidates bytes: {stats['total_token_candidates_bytes'] / 1024:.2f} KB")
    print(f"Total embedding bytes: {stats['total_embedding_bytes'] / 1024**2:.2f} MB")
    print(f"Total draft logits bytes: {stats['total_logits_bytes'] / 1024**2:.2f} MB")
    print(f"Total verify response bytes: {stats['total_verify_bytes'] / 1024**2:.2f} MB")

    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    print("\n=== Generated ===")
    print(generated_text)
    print("=================")

    del draft, output
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
