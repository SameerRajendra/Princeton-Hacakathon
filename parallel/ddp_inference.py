import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

# Reuse your existing setup_model from the benchmark script
from benchmarks.llama_spectre_inference_profile import setup_model


def init_dist(backend="nccl"):
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def run_inference_ddp(prompt: str, max_new_tokens: int = 32):
    rank, world_size = init_dist()
    device = torch.device(f"cuda:{rank}")

    # Each rank builds the same model
    model = setup_model(
        model_id     = "meta-llama/Llama-3.2-1B",
        seq_len      = 32768,
        weights_path = "llama32_spectre_final_weights.pt",
    ).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)
    model.eval()

    # Only rank 0 tokenizes, then broadcast to others
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
    else:
        input_ids = torch.zeros(1, 16, dtype=torch.long, device=device)

    dist.broadcast(input_ids, src=0)

    # Each rank runs the same single prompt (for latency measurement)
    with torch.no_grad():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()

        generated = input_ids
        for _ in range(max_new_tokens):
            out = model(input_ids=generated, use_cache=False)
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

        end.record()
        torch.cuda.synchronize()

    # Only rank 0 prints
    if rank == 0:
        ms = start.elapsed_time(end)
        total_new = generated.shape[1] - input_ids.shape[1]
        tok_per_s = total_new / (ms / 1000.0)
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated {total_new} tokens in {ms:.1f} ms "
              f"({tok_per_s:.0f} tok/s across 8 GPUs)")
        print(text)

    dist.destroy_process_group()


if __name__ == "__main__":
    run_inference_ddp(
        prompt="Explain the theory of relativity in simple terms:",
        max_new_tokens=32,
    )