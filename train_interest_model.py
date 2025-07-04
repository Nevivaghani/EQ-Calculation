# import json
# from datasets import load_dataset, Dataset
# from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
# import re

# # === Load your data ===
# def load_jsonl(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#     return [json.loads(line.strip()) for line in lines]

# data = load_jsonl("11interest_train_data.jsonl")
# # data = load_jsonl("interest_train_data.jsonl")

# # === Convert to HuggingFace Dataset ===
# dataset = Dataset.from_list(data)

# # === Load tokenizer and model ===
# model_name = "distilbert/distilgpt2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token  # avoid padding warning

# # === Tokenize ===
# def tokenize(batch):
#     # Use the full prompt and completion fields for training
#     label = batch.get("completion", "")
#     return tokenizer(
#         batch["prompt"] + "\n" + label,
#         truncation=True,
#         padding="max_length",
#         max_length=256
#     )

# tokenized_dataset = dataset.map(tokenize)

# # === Model ===
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # === Training Args ===
# training_args = TrainingArguments(
#     output_dir="./interest_model",
#     overwrite_output_dir=True,
#     num_train_epochs=1,
#     per_device_train_batch_size=5,
#     save_steps=100,
#     save_total_limit=2,
#     logging_steps=10,
#     prediction_loss_only=True,
#     learning_rate=5e-5,
# )

# # === Trainer Setup ===
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
#     data_collator=data_collator,
# )

# # === Train ===
# trainer.train()
# model.save_pretrained("./interest_model")
# tokenizer.save_pretrained("./interest_model")
# print("✅ Model trained and saved at ./interest_model")







import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch

# === Load JSONL data ===
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f.readlines()]

data = load_jsonl("11interest_train_data.jsonl")
dataset = Dataset.from_list(data)

# === Model: GPT-Neo 125M ===
model_name = "EleutherAI/gpt-neo-125M"

# === Tokenizer & Model ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Prevent warnings

model = AutoModelForCausalLM.from_pretrained(model_name)

# === Tokenize ===
def tokenize(example):
    text = example["prompt"] + "\n" + example.get("Interest Fields", "")
    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=2048
    )

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# === Data collator for CausalLM ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="./gptneo_interest_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    save_strategy="epoch",
    learning_rate=5e-5,
    logging_steps=10,
    report_to="none",
    eval_strategy="no",
    fp16=False,  # CPU only
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# === Train ===
trainer.train()

# === Save final model ===
model.save_pretrained("./gptneo_interest_model")
tokenizer.save_pretrained("./gptneo_interest_model")

print("✅ Model fine-tuned and saved to ./gptneo_interest_model")
