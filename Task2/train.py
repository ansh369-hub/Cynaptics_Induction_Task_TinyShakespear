import os
import torch
import gc
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from dataloader import load_alpaca_dataset


MODEL_NAME = "openai-community/gpt2-large"
OUTPUT_DIR = "./gpt2-large-alpaca-lora"
MAX_LENGTH = 512       
BATCH_SIZE = 16        
EPOCHS = 3
LEARNING_RATE = 2e-4   
LORA_R = 16
LORA_ALPHA = 32

def format_alpaca(example):
    """Formats the raw Alpaca dictionary into the standard instruction template."""
    if example.get("input"):
        text = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return {"text": text}

def main():
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"--- Initializing Training on {device} ---")

    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        dtype=torch.bfloat16, 
        device_map="auto"
    )

    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.1,
        target_modules=["c_attn"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    
    print("Loading and preprocessing Alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.map(format_alpaca)

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=MAX_LENGTH, 
            padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset.set_format("torch")
    train_dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)

    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    num_training_steps = len(train_dataloader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_dataloader):.4f}")

    
    print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training Complete!")

if __name__ == "__main__":
    main()

