from run_eq_flow import run_quiz_and_compute_eq
from predict_interest import load_eq_result, create_prompt, predict_interests

# 1️⃣ Run quiz + compute EQ + save result
run_quiz_and_compute_eq()

# 2️⃣ Predict interest fields from result
eq_data = load_eq_result("computed_eq_result.json")
prompt = create_prompt(eq_data)
interests = predict_interests(prompt)

print("\n🎯 FINAL SUGGESTED INTERESTS:", interests)
