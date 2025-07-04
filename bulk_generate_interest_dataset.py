import json
import os
import random
from compute_eq import compute_eq_and_traits

# Config
NUM_SAMPLES = 100
NUM_QUESTIONS = 10
AGE_GROUP = "11-15"
QUESTION_FILE = "cleaned_eq_dataset.json"

TRAIT_NAME_MAP = {
    "Self-Awareness": "Self-Awareness",
    "Self-Regulation": "Self-Regulation",
    "Motivation": "Motivation",
    "Motivatoin": "Motivation",
    "Empathy": "Empathy",
    "Social Skills": "Social Skills"
}

TRAIT_TO_INTEREST = {
    "Self-Awareness": ["Writing", "Journaling", "Philosophy", "Mind Mapping", "Self-documentary"],
    "Self-Regulation": ["Meditation", "Yoga", "Chess", "Gardening", "Martial Arts"],
    "Motivation": ["Entrepreneurship", "Public Speaking", "Science Projects", "Debates", "Leadership roles"],
    "Empathy": ["Teaching", "Animal Care", "Storytelling", "Nursing", "Conflict Mediation"],
    "Social Skills": ["Acting", "Team Sports", "Event Hosting", "Group Projects", "Community Work"]
}

with open(QUESTION_FILE, "r", encoding="utf-8") as f:
    all_questions = json.load(f)
question_pool = [q for q in all_questions if q["age_group"] == AGE_GROUP]

pretty_data = []
json_pretty_path = "11interest_train_data_pretty.json"
jsonl_path = "11interest_train_data.jsonl"

if os.path.exists(json_pretty_path):
    with open(json_pretty_path, "r", encoding="utf-8") as f:
        pretty_data = json.load(f)
        if not isinstance(pretty_data, list):
            pretty_data = [pretty_data]

for i in range(NUM_SAMPLES):
    random.shuffle(question_pool)
    selected = question_pool[:NUM_QUESTIONS]

    user_responses = []
    prompt_lines = [
        "User Query: What interests suit me based on my emotional responses?",
        f"Age Group: {AGE_GROUP}",
        "",
        "Responses:"
    ]

    for idx, q in enumerate(selected, 1):
        chosen_option = random.choice(q["options"])
        user_responses.append({
            "question_text": q["question_text"],
            "selected_label": chosen_option["label"],
            "trait": q["trait"],
            "trait_level": chosen_option["trait_level"],
            "age_group": q["age_group"]
        })

        prompt_lines.append(
            f"{idx}. Q: {q['question_text']}\n"
            f"   A: {chosen_option['label']}\n"
            f"   Trait: {TRAIT_NAME_MAP.get(q['trait'], q['trait'])}\n"
            f"   Trait Level: {chosen_option['trait_level']}"
        )

    # Compute EQ
    trait_summary, eq_score = compute_eq_and_traits(user_responses)

    prompt_lines.append("")
    prompt_lines.append(f"EQ Score: {round(eq_score, 2)}")
    prompt_lines.append("")
    prompt_lines.append("Trait Summary:")

    for trait, summary in trait_summary.items():
        fixed_trait = TRAIT_NAME_MAP.get(trait, trait)
        prompt_lines.append(f"- {fixed_trait} (Score: {round(summary['average_score'], 2)})")

    prompt = "\n".join(prompt_lines)

    # Determine Interests
    sorted_traits = sorted(trait_summary.items(), key=lambda x: x[1]["average_score"], reverse=True)
    top_traits = sorted_traits[:3]

    interests = []
    for trait, summary in top_traits:
        fixed_trait = TRAIT_NAME_MAP.get(trait, trait)
        if fixed_trait in TRAIT_TO_INTEREST:
            score = summary["average_score"]
            options = TRAIT_TO_INTEREST[fixed_trait]

            if score >= 4:
                interests.extend(options[:3])
            elif score >= 3:
                interests.extend(options[:2])
            else:
                interests.extend(options[:1])

    interests = sorted(set(interests))[:5]
    completion = "Interest Fields: " + ", ".join(interests)

    sample = {
        "prompt": prompt.strip(),
        "completion": completion.strip()
    }

    # Save to .jsonl
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Save to pretty JSON
    pretty_data.append(sample)

# Save pretty JSON file
with open(json_pretty_path, "w", encoding="utf-8") as f:
    json.dump(pretty_data, f, indent=2, ensure_ascii=False)

print(f"✅ Generated and saved {NUM_SAMPLES} full-context training samples.")
