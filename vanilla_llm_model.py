import torch
import torch.nn as nn
import time
import os
import deepspeed
from transformers import AutoConfig, AutoModelForCausalLM

# 1. Distributed Environment Initialization
deepspeed.init_distributed()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)

# 2. Config & Context Settings (Simulating Hackathon Scale)
MODEL_ID = "meta-llama/Llama-3.1-8B"
SEQUENCE_LENGTH = 32768  # Benchmark at 32K or 128K context to show the OOM cliff
BATCH_SIZE = 1           # Micro-batch size per GPU

print(f"[{local_rank}] Loading configuration for {MODEL_ID}...")
config = AutoConfig.from_pretrained(MODEL_ID)
config.use_cache = False  # Disabled to accurately measure backward pass activation sizes

# Force FlashAttention-2 as the standard optimization baseline
config._attn_implementation = "flash_attention_2" 

# 3. DeepSpeed ZeRO-3 Engine Configuration
ds_config = {
    "train_batch_size": BATCH_SIZE * torch.distributed.get_world_size(),
    "train_micro_batch_size_per_gpu": BATCH_SIZE,
    "zero_optimization": {
        "stage": 3,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
    },
    "bf16": {"enabled": True}, # Essential for H100 Hopper Tensor Cores
    "zero_allow_untested_optimizer": True
}

# 4. Initialize Model Under DeepSpeed Zero-3 Memory Sharding
with deepspeed.zero.Init():
    model = AutoModelForCausalLM.from_config(config)

# Setup dummy inputs mimicking packed FineWeb data
input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQUENCE_LENGTH), dtype=torch.long, device=device)
labels = input_ids.clone()

# Bind model to optimizer and deepspeed orchestration loop
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
model_engine, optimizer, _, _ = deepspeed.initialize(
    config=ds_config,
    model=model,
    optimizer=optimizer
)

# Warmup step to eliminate initial CUDA allocation overhead from metrics
print(f"[{local_rank}] Running initial warmup step...")
outputs = model_engine(input_ids=input_ids, labels=labels)
loss = outputs.loss
model_engine.backward(loss)
model_engine.step()
torch.cuda.synchronize()

# 5. Targeted Profiling Loop (Iterate 5 times for statistical variance)
print(f"[{local_rank}] Starting main benchmark runs...")
for iteration in range(5):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    # --- TARGETED NNSIGHT METRIC ZONE ---
    torch.cuda.nvtx.range_push(f"Llama3.1_8B_Step_{iteration}")
    
    # Forward Pass Tracking
    torch.cuda.nvtx.range_push("Forward_Pass")
    outputs = model_engine(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    torch.cuda.nvtx.range_pop() # Pop Forward_Pass
    
    # Backward Pass Tracking
    torch.cuda.nvtx.range_push("Backward_Pass")
    model_engine.backward(loss)
    torch.cuda.nvtx.range_pop() # Pop Backward_Pass
    
    # Optimizer Weight Update Tracking
    torch.cuda.nvtx.range_push("Optimizer_Step")
    model_engine.step()
    torch.cuda.nvtx.range_pop() # Pop Optimizer_Step
    
    torch.cuda.nvtx.range_pop() # Pop Llama3.1_8B_Step
    # -------------------------------------
    
    torch.cuda.synchronize()
    duration = time.perf_counter() - start_time
    tokens_processed = SEQUENCE_LENGTH * BATCH_SIZE * torch.distributed.get_world_size()
    throughput = tokens_processed / duration
    
    if local_rank == 0:
        print(f"Iteration {iteration} | Duration: {duration:.4f}s | Throughput: {throughput:.2f} tokens/sec")
        print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")