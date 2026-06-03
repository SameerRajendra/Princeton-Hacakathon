# ~/LLM_suite/parallel/tensor_parallel.py
# Splits SpectreMultiHead across 8 GPUs using PyTorch DistributedDataParallel
# + manual head sharding (Tensor Parallelism style, like Megatron-LM)

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from spectre import SpectreMultiHead


def init_process_group(backend="nccl"):
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


class TensorParallelSpectreMultiHead(nn.Module):
    """
    Splits num_heads across world_size GPUs.
    Each GPU owns (num_heads / world_size) heads.
    All-reduce over head outputs after concat.

    This mirrors Megatron-LM column/row parallel linear pattern
    but for SPECTRE heads.

    Comm volume per token:
      - All-reduce of (B, N, embed_dim) = 2 * B * N * embed_dim * 2 bytes (bf16)
      - At B=1, N=32k, d=2048: ~256 MB per layer — within NVLink bandwidth
    """

    def __init__(self, embed_dim, num_heads, n_fft, rank, world_size, **kwargs):
        super().__init__()
        assert num_heads % world_size == 0, \
            f"num_heads ({num_heads}) must be divisible by world_size ({world_size})"

        self.rank = rank
        self.world_size = world_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.heads_per_gpu = num_heads // world_size

        # Each GPU only holds its local head slice
        self.local_heads = nn.ModuleList([
            SpectreMultiHead(
                embed_dim   = embed_dim,
                num_heads   = self.heads_per_gpu,
                n_fft       = n_fft,
                **kwargs
            )
        ])

        # Row-parallel output projection (each GPU has a slice)
        self.out_proj = nn.Linear(
            embed_dim // world_size * self.heads_per_gpu,
            embed_dim,
            bias=False
        )

    def forward(self, x):
        # x: (B, N, embed_dim) — full on each GPU
        # Slice the embedding dimension for this GPU's heads
        d_per_rank = self.embed_dim // self.world_size
        x_local = x[..., self.rank * d_per_rank:(self.rank + 1) * d_per_rank]

        # Local head computation
        out_local = self.local_heads[0](x_local)       # (B, N, d_local)
        out_proj   = self.out_proj(out_local)           # (B, N, embed_dim)

        # All-reduce to sum partial outputs across GPUs (row-parallel pattern)
        dist.all_reduce(out_proj, op=dist.ReduceOp.SUM)
        return out_proj


def wrap_model_tensor_parallel(model, rank, world_size):
    """
    Replace all SpectreMultiHead modules in model with
    TensorParallelSpectreMultiHead equivalents.
    Called once per GPU process after model init.
    """
    for name, module in model.named_children():
        if isinstance(module, SpectreMultiHead):
            tp_module = TensorParallelSpectreMultiHead(
                embed_dim    = module.head_dim * module.num_heads,
                num_heads    = module.num_heads,
                n_fft        = module.heads[0].n_fft,
                rank         = rank,
                world_size   = world_size,
                d_gate       = module.heads[0].gate_mlp[0].out_features,
                wavelet_on_rate = module.wavelet_refinement.on_rate,
            ).to(f"cuda:{rank}")
            setattr(model, name, tp_module)
        else:
            wrap_model_tensor_parallel(module, rank, world_size)
    return model