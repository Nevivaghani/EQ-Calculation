import json
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

# === Load your data ===
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [json.loads(line.strip()) for line in lines]

data = load_jsonl("11interest_train_data.jsonl")

# === Convert to HuggingFace Dataset ===
dataset = Dataset.from_list(data)

# === Load tokenizer and model ===
model_name = "distilbert/distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # avoid padding warning

# === Tokenize ===
def tokenize(batch):
    return tokenizer(
        batch["prompt"] + "\n" + batch["completion"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize)

# === Model ===
model = AutoModelForCausalLM.from_pretrained(model_name)

# === Training Args ===
training_args = TrainingArguments(
    output_dir="./interest_model",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    prediction_loss_only=True,
    learning_rate=5e-5,
    report_to="none"
)

# === Trainer Setup ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# === Train ===
trainer.train()
model.save_pretrained("./interest_model")
tokenizer.save_pretrained("./interest_model")
print("✅ Model trained and saved at ./interest_model")
