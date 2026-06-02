import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def build_fineweb_dataloader(
    model_id="meta-llama/Llama-3.2-1B",
    dataset_name="HuggingFaceFW/fineweb",
    subset="sample-10BT", # Use a sample subset for development
    seq_length=8192,      # Adjust based on your SPECTRE context window (e.g., 32768)
    batch_size=4
):
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading FineWeb dataset in streaming mode...")
    # Streaming is essential since FineWeb is extremely large
    dataset = load_dataset(dataset_name, name=subset, split="train", streaming=True)

    def tokenize_function(examples):
        # We don't truncate or pad here; we do that in the packing step
        return tokenizer(examples["text"], truncation=False, padding=False)

    def group_texts(examples):
        # Concatenate all tokenized texts in the batch
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Drop the small remainder that doesn't fit into our exact seq_length
        if total_length >= seq_length:
            total_length = (total_length // seq_length) * seq_length
            
        # Split by chunks of seq_length
        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated_examples.items()
        }
        # For Causal Language Modeling, labels are the input_ids
        result["labels"] = result["input_ids"].copy()
        return result

    # Remove original text columns to save memory
    columns_to_remove = ["text", "id", "dump", "url", "date", "file_path", "language", "language_score", "token_count"]
    
    print("Applying tokenization and packing...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)
    packed_dataset = tokenized_dataset.map(group_texts, batched=True)

    def collate_fn(batch):
        # Convert list of dicts to dict of tensors
        return {
            "input_ids": torch.tensor([item["input_ids"] for item in batch]),
            "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
            "labels": torch.tensor([item["labels"] for item in batch]),
        }

    dataloader = DataLoader(
        packed_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    
    return dataloader, tokenizer

if __name__ == "__main__":
    # Test the dataloader instantiation
    loader, tok = build_fineweb_dataloader(seq_length=8192, batch_size=2)
    
    # Fetch the first batch to verify
    for batch in loader:
        print("Batch input_ids shape:", batch["input_ids"].shape)
        break