import os
os.environ["TRANSFORMERS_NO_TF"] = "1"


import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# =========================
# CONFIG
# =========================

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Please read the problem carefully before writing any Python code."
)

DATASET_NAME = "deep"   # "deep" OR "diverse"
OUTPUT_DIR = f"./lora_{DATASET_NAME}"

MAX_LENGTH = 1024
EPOCHS = 3
LR = 2e-4
BATCH_SIZE = 2
GRAD_ACC = 8
SEED = 42

# =========================
# 1️⃣ BASELINE INFERENCE
# =========================

def baseline_inference():
    print("\n🔹 Running baseline inference...\n")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    prompt = f"""{SYSTEM_PROMPT}

Problem:
Write a Python function to check if a number is prime.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# =========================
# 2️⃣ LOAD DATASET
# =========================

def load_codegen_dataset():
    if DATASET_NAME == "deep":
        ds = load_dataset("Naholav/CodeGen-Deep-5K")
    else:
        ds = load_dataset("Naholav/CodeGen-Diverse-5K")

    # Train / Val / Test split
    split_1 = ds["train"].train_test_split(test_size=0.1, seed=SEED)
    split_2 = split_1["test"].train_test_split(test_size=0.5, seed=SEED)

    return {
        "train": split_1["train"],
        "validation": split_2["train"],
        "test": split_2["test"]
    }


# =========================
# 3️⃣ TOKENIZATION (solution-only)
# =========================

def preprocess(example):
    prompt = f"""{SYSTEM_PROMPT}

Problem:
{example['input']}

Solution:
"""
    text = prompt + example["solution"]

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


# =========================
# 4️⃣ TRAINING
# =========================

def train_lora(dataset):

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized = {
        k: dataset[k].map(
            preprocess,
            remove_columns=dataset[k].column_names
        )
        for k in dataset
    }

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        evaluation_strategy="steps",
        save_steps=200,
        eval_steps=100,
        logging_steps=40,
        learning_rate=LR,
        fp16=True,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to="none",
        save_total_limit=3
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
    )

    trainer.train()

    return tokenized["test"]


# =========================
# 5️⃣ CHECKPOINT EVALUATION
# =========================

def evaluate_checkpoints(test_dataset):
    best_loss = float("inf")
    best_ckpt = None

    for ckpt in os.listdir(OUTPUT_DIR):
        if "checkpoint" not in ckpt:
            continue

        ckpt_path = os.path.join(OUTPUT_DIR, ckpt)
        print(f"\n🔍 Evaluating {ckpt_path}")

        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        total_loss = 0
        for ex in test_dataset:
            inputs = tokenizer(
                ex["solution"],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])

            total_loss += outputs.loss.item()

        avg_loss = total_loss / len(test_dataset)
        print(f"📉 Avg Test Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = ckpt_path

    print("\n🏆 BEST CHECKPOINT")
    print(best_ckpt)
    print("Loss:", best_loss)


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    baseline_inference()

    dataset = load_codegen_dataset()

    test_set = train_lora(dataset)

    evaluate_checkpoints(test_set)
