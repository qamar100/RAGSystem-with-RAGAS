import json


from GenerateData import qa_pairs

# Format for RAGAS without contexts
ragas_dataset = [
    {
        "question": qa_pair["question"],
        "ground_truths": [qa_pair["answer"]]
    }
    for qa_pair in qa_pairs
]

# Save RAGAS-formatted dataset
with open("ragas_dataset.json", "w") as f:
    json.dump(ragas_dataset, f, indent=2)

print("RAGAS-formatted dataset saved to ragas_dataset.json")