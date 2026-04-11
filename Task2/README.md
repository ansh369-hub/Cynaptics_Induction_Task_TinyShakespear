***

# Task 2: Supervised Fine-Tuning of GPT-2 on Alpaca (Optional Task)

**Fine-Tuning**. Your objective is to take a pretrained GPT-2 base model and fine-tune it on the Stanford Alpaca dataset to follow instructions in a conversational assistant style.

Unlike Task 1 where you built and pretrained a model from scratch, here you will leverage the power of **transfer learning** — starting from a model that already understands language and adapting it to a new downstream task.

## 📋 Task 2 Overview

You will load the pretrained **GPT-2 base (124M)** model from Hugging Face and perform **Supervised Fine-Tuning (SFT)** on the Alpaca instruction-following dataset. By the end, your model should be able to take an instruction (and optional input) and generate a helpful assistant-style response.

### Pretrained Model
* **Hugging Face Link:** [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)

### Dataset
* **Alpaca Dataset:** [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)

The Alpaca dataset contains ~52K instruction-following examples, each with:
- `instruction` — The task description
- `input` — Optional additional context  
- `output` — The expected assistant response

### Core Deliverables:

**All deliverable code should be python files, other formats like jupyter notebooks are not allowed**.

1. **Dataset Preparation:** Load and format the Alpaca dataset into a prompt template suitable for causal language model fine-tuning.
2. **Model Loading:** Load the pretrained GPT-2 base model and tokenizer from Hugging Face.
3. **Fine-Tuning Loop:** Implement a training loop to fine-tune the model with Cross-Entropy Loss (or any other loss).
4. **Inference:** A generation script that takes an instruction prompt and generates an assistant-style response.

## 💬 Prompt Template

Use the following prompt format to structure each training example (given in alpaca):

**With input context:**
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

**Without input context:**
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}
```
---

## 🚀 Quick Start

1. Install Dependencies

Bash
pip install torch transformers datasets peft tqdm
2. Run the Training Script
(Note: Requires a GPU. The default configuration is optimized for an A100 or similar high-VRAM machine).

Bash
python train.py
3. Run Inference

Bash
python inference.py

---

## 🏗️ Final Code Architecture

To move beyond basic pattern mimicry and achieve actual instructional comprehension, the final architecture relies on a highly optimized pipeline:

Base Model: openai-community/gpt2-large (774M parameters). Chosen over the base 124M model to give the network enough capacity to "reason" through complex instructions.

Adapter: LoRA (Low-Rank Adaptation) via the peft library. Targets the c_attn layers with a rank (r) of 16.

Precision: Native PyTorch bfloat16.

Optimizer & Scheduler: AdamW combined with a Cosine Annealing Scheduler (10% warmup) to prevent catastrophic forgetting and stabilize the loss curve early in training.

Inference Pipeline: The inference script loads the frozen gpt2-large base model and dynamically applies the saved LoRA adapter, using Nucleus Sampling (top_p=0.9) and a repetition penalty (1.2) to prevent classic GPT-2 looping.


## 🧠 Technical Nuances & Implementation Details

During development, several PyTorch/Hugging Face quirks had to be addressed:

The Padding Token Trap: GPT-2 does not have a default padding token. I had to manually set tokenizer.pad_token = tokenizer.eos_token and map it to the model config to ensure batched inputs didn't crash.

Causal LM Label Shifting: In SFT, the model learns to predict the next token. Hugging Face's GPT2LMHeadModel automatically shifts logits internally. Therefore, the data pipeline explicitly sets labels = input_ids.

Prompt Format Slicing: Because GPT-2 is a decoder-only causal model, it generates the prompt alongside the answer. My inference.py script includes custom slicing logic to extract and return only the newly generated text following the ### Response:\n marker.


## 💥 Errors, Roadblocks & Learning Experiences

Phase 1: The 2-Hour Bottleneck
Initially, I tried to train the base gpt2 model on the full 52k dataset using standard fp32 precision. To meet the time constraints, I had to implement "survival tactics": truncating sequence lengths to 256 and subsetting the dataset to 5,000 rows. While it trained fast, the model was starved of context and data.

Phase 2: The Final Boss (CUDA Out of Memory)
When I attempted to scale up to gpt2-large and increase the sequence length to 1024 for better response quality, I immediately hit a wall:

CUDA out of memory. Tried to allocate 786.00 MiB. GPU 0 has a total capacity of 14.56 GiB... If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True

The Solution:
This OOM error forced me to completely rethink my memory footprint. I learned that simply clearing cache wasn't enough. I refactored the pipeline to include:

Memory De-fragmentation: Implemented the expandable_segments:True environment variable.

Mixed Precision: Shifted to bfloat16, practically halving the memory footprint.

LoRA: Instead of updating 774 million parameters, I froze the base model and trained less than 1% of the weights via a LoRA adapter. This entirely eliminated my VRAM bottleneck and allowed me to comfortably train on an A100.

---

## 📝 Sample Outputs
(After running inference.py, paste your 3 required sample outputs here to prove the model works!)

Test 1: Instruction without Input

Instruction: What is a digital identity and why is it important?

Assistant: [Paste output here]

Test 2: Instruction with Input

Instruction: Summarize the following text.

Input: Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass more than two and a half times that of all the other planets in the Solar System combined, but slightly less than one-thousandth the mass of the Sun.

Assistant: [Paste output here]

Test 3: Custom Generation

Instruction: [Write your own instruction]

Assistant: [Paste output here]