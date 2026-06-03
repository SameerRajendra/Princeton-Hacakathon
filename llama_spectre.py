import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from datasets import load_dataset
from torch.optim import AdamW
from accelerate import Accelerator
import pandas as pd

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
            use_toeplitz=False, 
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
        
        return out

# -----------------------------------------------------------------------
# 2. Model Initialization
# -----------------------------------------------------------------------
def setup_model(model_id="meta-llama/Llama-3.2-1B", seq_len=1024):
    print(f"Loading base config for {model_id}...")
    config = AutoConfig.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    
    print(f"Swapping standard Llama attention layers for SpectreBlocks (n_fft={seq_len})...")
    for i in range(len(model.model.layers)):
        model.model.layers[i] = SpectreLlamaLayerWrapper(config, n_fft=seq_len)
        
    return model

# -----------------------------------------------------------------------
# 3. Distributed Training Loop (Step-Based)
# -----------------------------------------------------------------------
def main():
    accelerator = Accelerator()
    
    seq_len = 1024  
    batch_size = 2       # Batch size PER GPU
    max_steps = 10    # <--- SET YOUR EXACT TARGET STEPS HERE
    
    model = setup_model(seq_len=seq_len)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    # Wrap model and optimizer for distributed training
    model, optimizer = accelerator.prepare(model, optimizer)
    
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        name="sample-10BT", 
        split="train", 
        streaming=True
    )
    
    model.train()
    metrics_log = []
    
    if accelerator.is_main_process:
        print(f"Starting distributed training on {accelerator.num_processes} GPUs.")
        print(f"Training for exactly {max_steps} steps.")
        print("-" * 65)
    
    for step, batch in enumerate(dataset):
        # 1. Step Limit Check
        if step >= max_steps:
            if accelerator.is_main_process:
                print(f"\nReached {max_steps} steps. Halting loop.")
            break
            
        # 2. Tokenize and push to specific GPU
        tokens = tokenizer(
            batch["text"], 
            max_length=seq_len, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        input_ids = tokens.input_ids.repeat(batch_size, 1).to(accelerator.device)
        labels = input_ids.clone()
        
        # 3. Forward & Backward Pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        # 4. Logging (Only on Main GPU to avoid terminal spam)
        if step % 50 == 0 and accelerator.is_main_process:
            print(f"Step {step:<8} / {max_steps} | Loss: {loss.item():<8.4f}")
            metrics_log.append({
                "Step": step, 
                "Loss": loss.item()
            })

    # -----------------------------------------------------------------------
    # 4. Save State Checkpoint
    # -----------------------------------------------------------------------
    # CRITICAL: Make sure no GPU runs ahead and tries to save before the others finish
    accelerator.wait_for_everyone() 
    
    if accelerator.is_main_process:
        print("\nSaving metrics to CSV...")
        pd.DataFrame(metrics_log).to_csv("spectre_distributed_metrics.csv", index=False)
        
        print("Unwrapping model and saving weights...")
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), "llama32_spectre_final_weights.pt")
        print("Training successfully complete and model saved.")

if __name__ == "__main__":
    main()