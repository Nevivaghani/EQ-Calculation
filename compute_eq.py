import json
from collections import defaultdict

# Mapping trait level to numeric score
TRAIT_LEVEL_MAP = {
    "Very Low": 1,
    "Low": 2,
    "Moderate": 3,
    "High": 4,
    "Very High": 5
}

def compute_eq_and_traits(user_responses):
    trait_scores = defaultdict(list)

    for entry in user_responses:
        trait = entry["trait"]
        level = entry["trait_level"]
        score = TRAIT_LEVEL_MAP.get(level, 0)
        trait_scores[trait].append(score)

    trait_summary = {}
    for trait, scores in trait_scores.items():
        trait_summary[trait] = {
            "average_score": round(sum(scores) / len(scores), 2),
            "raw_scores": scores
        }

    # Compute total EQ score (normalize to 100)
    total_avg = sum(t["average_score"] for t in trait_summary.values()) / len(trait_summary)
    eq_score = round((total_avg / 5) * 100, 2)  # Normalize: max 5 → 100

    return trait_summary, eq_score

# Optional standalone testing
if __name__ == "__main__":
    with open("user_15_responses.json", "r", encoding="utf-8") as f:
        responses = json.load(f)

    trait_summary, eq_score = compute_eq_and_traits(responses)

    result = {
        "eq_score": eq_score,
        "trait_summary": trait_summary,
        "age_group": responses[0]["age_group"] if responses else "unknown"
    }

    print(json.dumps(result, indent=2))

    with open("computed_eq_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("✅ EQ and trait summary saved to computed_eq_result.json")
