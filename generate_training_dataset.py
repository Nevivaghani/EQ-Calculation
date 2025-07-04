import json
import os
import random

# === Fix trait typos or inconsistencies ===
TRAIT_NAME_MAP = {
    "Self-Awareness": "Self-Awareness",
    "Self-Regulation": "Self-Regulation",
    "Motivation": "Motivation",
    "Motivatoin": "Motivation",  # typo fix
    "Empathy": "Empathy",
    "Social Skills": "Social Skills"
}

# === Interest mapping by trait ===
TRAIT_TO_INTEREST = {
    "Self-Awareness": ["Writing", "Journaling", "Philosophy"],
    "Self-Regulation": ["Meditation", "Yoga", "Chess"],
    "Motivation": ["Entrepreneurship", "Competitive Sports", "Public Speaking"],
    "Empathy": ["Teaching", "Nursing", "Counseling"],
    "Social Skills": ["Acting", "Team Sports", "Debating"]
}

# === Load EQ result JSON ===
with open("computed_eq_result.json", "r", encoding="utf-8") as f:
    result = json.load(f)

eq_score = result["eq_score"]
age_group = result["age_group"]
traits = result["trait_summary"]

# === Build prompt text ===
prompt_lines = [
    f"EQ Score: {eq_score}",
    "Trait Summary:"
]
for trait, summary in traits.items():
    fixed_trait = TRAIT_NAME_MAP.get(trait, trait)
    prompt_lines.append(f"- {fixed_trait}: {summary['average_score']}")
prompt_lines.append(f"Age Group: {age_group}")
prompt = "\n".join(prompt_lines)

# === Select interests from top 3 scoring traits ===
sorted_traits = sorted(traits.items(), key=lambda x: x[1]["average_score"], reverse=True)
top_traits = sorted_traits[:3]

interests = []
for trait, summary in top_traits:
    fixed_trait = TRAIT_NAME_MAP.get(trait, trait)
    if fixed_trait in TRAIT_TO_INTEREST:
        level = summary["average_score"]
        if level >= 3.5:
            interests.extend(random.sample(TRAIT_TO_INTEREST[fixed_trait], k=2))
        elif level >= 2.5:
            interests.extend(random.sample(TRAIT_TO_INTEREST[fixed_trait], k=1))

# === Finalize interest list ===
interests = list(set(interests))
random.shuffle(interests)
interests = interests[:5]
target = "Interest Fields: " + ", ".join(interests)

# === Format training sample ===
sample = {
    "prompt": prompt.strip(),
    "completion": target.strip()
}

# === Append to .jsonl ===
jsonl_path = "interest_train_data.jsonl"
with open(jsonl_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# === Append to .json list ===
json_pretty_path = "interest_train_data_pretty.json"
if os.path.exists(json_pretty_path):
    with open(json_pretty_path, "r", encoding="utf-8") as f:
        pretty_data = json.load(f)
        if not isinstance(pretty_data, list):
            pretty_data = [pretty_data]
else:
    pretty_data = []

pretty_data.append(sample)

with open(json_pretty_path, "w", encoding="utf-8") as f:
    json.dump(pretty_data, f, indent=2, ensure_ascii=False)

print("✅ Appended new sample:")
print("  • interest_train_data.jsonl (multi-line)")
print("  • interest_train_data_pretty.json (list)")
