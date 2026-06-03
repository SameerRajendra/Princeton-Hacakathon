# ~/LLM_suite/parallel/launch_inference.py
# torchrun --nproc_per_node=8 parallel/launch_inference.py

import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from tensor_parallel import init_process_group, wrap_model_tensor_parallel
import sys
sys.path.insert(0, os.path.expanduser("~/LLM_suite"))
from benchmarks.llama_spectre_inference_profile import setup_model


def run(prompt: str, max_new_tokens: int = 50):
    rank, world_size = init_process_group()
    device = f"cuda:{rank}"

    model = setup_model(
        model_id     = "meta-llama/Llama-3.2-1B",
        seq_len      = 32768,
        weights_path = "llama32_spectre_final_weights.pt",
    )

    # Shard heads across 8 GPUs
    model = wrap_model_tensor_parallel(model, rank, world_size)
    model = model.to(device).eval()

    # Only rank 0 handles tokenization and output
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
    else:
        input_ids = torch.zeros(1, 16, dtype=torch.long, device=device)

    # Broadcast input_ids to all ranks
    dist.broadcast(input_ids, src=0)

    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

        generated = input_ids
        for _ in range(max_new_tokens):
            out  = model(input_ids=generated, use_cache=False)
            next = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next], dim=1)
            if next.item() == 2:  # EOS
                break

        t1.record()
        torch.cuda.synchronize()

    if rank == 0:
        ms = t0.elapsed_time(t1)
        toks = generated.shape[1] - input_ids.shape[1]
        print(f"Generated {toks} tokens in {ms:.1f} ms  "
              f"({toks / ms * 1000:.0f} tok/s)")
        print(tokenizer.decode(generated[0], skip_special_tokens=True))

    dist.destroy_process_group()


if __name__ == "__main__":
    run("Explain the theory of relativity in simple terms:", max_new_tokens=50)