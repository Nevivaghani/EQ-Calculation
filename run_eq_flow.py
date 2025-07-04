import json
import random
from compute_eq import compute_eq_and_traits

def run_quiz_and_compute_eq(age_group="8-10", num_questions=10, question_file="cleaned_eq_dataset.json"):
    # Load full question dataset
    with open(question_file, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    # Filter by age group
    filtered = [q for q in all_questions if q["age_group"] == age_group]
    random.shuffle(filtered)
    selected_questions = filtered[:num_questions]

    print(f"\n🧠 EQ Quiz — Age Group: {age_group}")
    user_responses = []

    for i, q in enumerate(selected_questions):
        print(f"\n🔹 Q{i+1}: {q['question_text']}")
        for idx, opt in enumerate(q["options"]):
            print(f"  {idx + 1}. {opt['label']}")

        while True:
            try:
                choice = int(input("👉 Select (1/2/3...): ")) - 1
                if 0 <= choice < len(q["options"]):
                    selected = q["options"][choice]
                    user_responses.append({
                        "question_text": q["question_text"],
                        "selected_label": selected["label"],
                        "trait": q["trait"],
                        "trait_level": selected["trait_level"],
                        "age_group": q["age_group"]
                    })
                    break
                else:
                    print("❌ Invalid option. Try again.")
            except ValueError:
                print("❌ Please enter a number.")

    # Calculate EQ and trait summary
    trait_summary, eq_score = compute_eq_and_traits(user_responses)

    # Final result
    result = {
        "eq_score": eq_score,
        "trait_summary": trait_summary,
        "age_group": age_group,
        "responses": user_responses
    }

    # Save to file
    with open("computed_eq_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n✅ EQ Summary:")
    print(json.dumps(result, indent=2))

# If running standalone
if __name__ == "__main__":
    run_quiz_and_compute_eq()
