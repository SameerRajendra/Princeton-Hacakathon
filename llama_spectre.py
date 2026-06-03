import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from datasets import load_dataset
import torch.cuda.nvtx as nvtx
from torch.optim import AdamW

# Import your native PyTorch SPECTRE block
from spectre import SpectreBlock

# -----------------------------------------------------------------------
# 1. Hugging Face Compatibility Wrapper
# -----------------------------------------------------------------------
class SpectreLlamaLayerWrapper(nn.Module):
    def __init__(self, config, n_fft):
        super().__init__()
        self.spectre = SpectreBlock(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            n_fft=n_fft,
            mlp_ratio=4,          
            use_toeplitz=False,   # Keep False for your dynamic injection
            pooling_type="dct",
            wavelet_on_rate=0.1
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs
    ):
        # SAFETY NET: Unpack just in case the first embedding layer hands us a tuple
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
            
        # 1. Capture the incoming dtype (BFloat16)
        orig_dtype = hidden_states.dtype
        
        # 2. Cast to Float32 for stable FFTs and complex math
        hidden_states_f32 = hidden_states.to(torch.float32)
        
        # 3. Process through the SPECTRE block
        out_f32 = self.spectre(hidden_states_f32)
        
        # 4. Cast back to the original dtype
        out = out_f32.to(orig_dtype)
        
        # 5. RETURN RAW TENSOR (Removed the tuple packaging)
        return out
# -----------------------------------------------------------------------
# 2. Model Initialization & Architecture Swap
# -----------------------------------------------------------------------
def setup_model(model_id="meta-llama/Llama-3.2-1B", seq_len=1024):
    print(f"Loading base config for {model_id}...")
    config = AutoConfig.from_pretrained(model_id)
    
    # Initialize randomly for training from scratch
    # Use .from_pretrained(model_id) if you are fine-tuning existing weights
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    
    print(f"Swapping standard Llama attention layers for SpectreBlocks (n_fft={seq_len})...")
    for i in range(len(model.model.layers)):
        model.model.layers[i] = SpectreLlamaLayerWrapper(config, n_fft=seq_len)
        
    model = model.to("cuda")
    return model

# -----------------------------------------------------------------------
# 3. Training Loop with Nsight Instrumentation
# -----------------------------------------------------------------------
def main():
    seq_len = 1024  # Ensure this matches your expected n_fft
    batch_size = 2  # Adjust based on your GPU VRAM
    
    model = setup_model(seq_len=seq_len)
    
    # Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    # Load FineWeb-edu (sample-10BT is excellent for fast streaming & testing)
    print("Streaming FineWeb-edu dataset...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        name="sample-10BT", 
        split="train", 
        streaming=True
    )
    
    # Warmup steps allow the PyTorch CUDA memory allocator to settle
    # This prevents profiling artifacts related to initial memory allocation
    warmup_steps = 3
    profile_steps = 5
    
    model.train()
    print("Starting training loop...")
    
    for step, batch in enumerate(dataset):
        # if step >= warmup_steps + profile_steps:
        #     break
            
        # Tokenize sequence
        tokens = tokenizer(
            batch["text"], 
            max_length=seq_len, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        input_ids = tokens.input_ids.repeat(batch_size, 1).to("cuda")
        labels = input_ids.clone()
        
        # Start Profiling Capture after warmup
        if step == warmup_steps:
            print("Warmup complete. Starting Nsight capture...")
            torch.cuda.cudart().cudaProfilerStart()

        # # NVTX Marker: Full Step
        # nvtx.range_push(f"Step_{step}")
        
        # # NVTX Marker: Forward Pass
        # nvtx.range_push("Forward_Pass")
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        # nvtx.range_pop() 

        # # NVTX Marker: Backward Pass
        # nvtx.range_push("Backward_Pass")
        loss.backward()
        # nvtx.range_pop() 

        # NVTX Marker: Optimizer
        # nvtx.range_push("Optimizer_Step")
        optimizer.step()
        optimizer.zero_grad()
        # nvtx.range_pop() 
        
        # nvtx.range_pop() # End Step
        
        print(f"Step {step} | Loss: {loss.item():.4f}")

    torch.cuda.cudart().cudaProfilerStop()
    print("Profiling complete.")

if __name__ == "__main__":
    main()