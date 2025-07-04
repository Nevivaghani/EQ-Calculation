import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Load trained model ===
model_path = "./interest_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# === Load actual computed_eq_result.json ===
def load_eq_result(path="computed_eq_result.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# === Create prompt using real trait_summary structure ===
def create_prompt(eq_data):
    eq_score = eq_data.get("eq_score", "")
    age_group = eq_data.get("age_group", "")
    responses = eq_data.get("responses", [])
    trait_summary = eq_data.get("trait_summary", {})

    lines = ["User Query: What interests suit me based on my emotional responses?"]
    lines.append(f"Age Group: {age_group}")
    lines.append("\nResponses:")

    for idx, r in enumerate(responses, 1):
        q = r["question_text"]
        a = r["selected_label"]
        trait = r["trait"]
        level = r["trait_level"]
        lines.append(f"{idx}. Q: {q}\n   A: {a}\n   Trait: {trait}\n   Trait Level: {level}")

    lines.append(f"\nEQ Score: {eq_score}")
    lines.append("\nTrait Summary:")
    for trait, data in trait_summary.items():
        avg = round(data.get("average_score", 0), 2)
        lines.append(f"- {trait} (Score: {avg})")  # ✅ critical fix: match (Score: x.xx)

    lines.append("\nInterest Fields:")
    return "\n".join(lines)

import re

def extract_interest_fields(full_output):
    # Look for "Interest Fields: ..." pattern
    match = re.search(r"Interest Fields:\s*(.+)", full_output)
    if match:
        return match.group(1).strip().split("\n")[0]  # Return only first line
    return full_output.strip()  # Fallback

# === Predict interest fields ===
def predict_interests(prompt, max_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.8,           # ← improved sampling
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id  # ← Important
    )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated = full_output.replace(prompt, "").strip()
    generated = generated.split("\n")[0].strip()

    # 🧹 Stop at first newline or junk
    # generated = generated.split("\n")[0].strip()
    return generated

# === Run everything ===
if __name__ == "__main__":
    eq_data = load_eq_result("computed_eq_result.json")
    prompt = create_prompt(eq_data)
    print("🔹 Prompt to model:\n", prompt)
    
    interests = predict_interests(prompt)
    print("\n🎯 Suggested Interests:", interests)
