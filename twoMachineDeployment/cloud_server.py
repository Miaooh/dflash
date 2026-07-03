import io
import time
import torch
import grpc
from concurrent import futures
from transformers import AutoModelForCausalLM, AutoTokenizer

import dflash_service_pb2
import dflash_service_pb2_grpc

from dflash.model import extract_context_feature, sample

TARGET_PATH = "/home/xzh/models/Qwen3-4B"
DEVICE = "cuda:0"
PORT = 50051
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


class DFlashCloudServicer(dflash_service_pb2_grpc.DFlashCloudServicer):
    def __init__(self, target, tokenizer):
        self.target = target
        self.tokenizer = tokenizer
        self.past_key_values = None

    def Prefill(self, request, context):
        input_ids = proto_to_tensor(request.input_ids, DEVICE)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)

        t0 = time.perf_counter()
        output = self.target(
            input_ids,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=True,
            output_hidden_states=True,
        )
        elapsed = time.perf_counter() - t0
        print(f"[Cloud] Prefill seq_len={seq_len}, time={elapsed:.3f}s")

        self.past_key_values = output.past_key_values
        target_hidden = extract_context_feature(output.hidden_states, self.target.config.dflash_target_layer_ids)
        first_token_logits = output.logits[:, -1:, :]

        return dflash_service_pb2.PrefillResponse(
            hidden_states=tensor_to_proto(target_hidden),
            first_token_logits=tensor_to_proto(first_token_logits),
        )

    def Verify(self, request, context):
        candidate_ids = proto_to_tensor(request.candidate_ids, DEVICE)
        position_ids = proto_to_tensor(request.position_ids, DEVICE)
        crop_to = request.crop_to

        if crop_to > 0 and self.past_key_values is not None:
            self.past_key_values.crop(crop_to)

        t0 = time.perf_counter()
        output = self.target(
            candidate_ids,
            position_ids=position_ids,
            past_key_values=self.past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        elapsed = time.perf_counter() - t0
        print(f"[Cloud] Verify candidates={candidate_ids.shape[1]}, crop_to={crop_to}, time={elapsed:.3f}s")

        self.past_key_values = output.past_key_values
        target_hidden = extract_context_feature(output.hidden_states, self.target.config.dflash_target_layer_ids)

        return dflash_service_pb2.VerifyResponse(
            logits=tensor_to_proto(output.logits),
            hidden_states=tensor_to_proto(target_hidden),
        )


def load_target_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH)

    print("Loading target model...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH,
        dtype="auto",
        device_map=DEVICE,
    ).eval()

    if not hasattr(target.config, "dflash_target_layer_ids"):
        from dflash.model import build_target_layer_ids
        num_target_layers = target.config.num_hidden_layers
        num_draft_layers = 5
        target.config.dflash_target_layer_ids = build_target_layer_ids(num_target_layers, num_draft_layers)
        print(f"Set dflash_target_layer_ids = {target.config.dflash_target_layer_ids}")

    return target, tokenizer


def serve():
    target, tokenizer = load_target_model()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=GRPC_OPTIONS)
    dflash_service_pb2_grpc.add_DFlashCloudServicer_to_server(
        DFlashCloudServicer(target, tokenizer), server
    )
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    print(f"[Cloud] Server started on port {PORT}")
    print("[Cloud] Waiting for edge client requests...")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[Cloud] Shutting down server...")
        server.stop(0)


if __name__ == "__main__":
    serve()
