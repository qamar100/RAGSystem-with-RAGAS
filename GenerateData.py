import openai
import json
import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def generate_hr_qa_pairs(num_pairs=50):
    all_qa_pairs = []
    
    for i in range(0, num_pairs, 10):  # Generate in batches of 10
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert generating questions and answers for [YOUR SPECIFIC USE CASE]."},
                {"role": "user", "content": f"Generate {min(10, num_pairs - i)} diverse [YOUR SPECIFIC USE CASE] questions and detailed answers."}
            ],
            temperature=0.7,
        )
        
        qa_text = response.choices[0].message.content
        qa_pairs = parse_qa_pairs(qa_text)
        all_qa_pairs.extend(qa_pairs)
    
    return all_qa_pairs

def parse_qa_pairs(text):
    pairs = []
    lines = text.split("\n")
    for i in range(0, len(lines), 3):  # Assuming Q and A are on separate lines
        if i+1 < len(lines):
            question = lines[i].strip()
            answer = lines[i+1].strip()
            # Remove prefix "Q" or "A" if it exists
            if question.lower().startswith("q"):
                question = question[question.find(":")+1:].strip()
            if answer.lower().startswith("a"):
                answer = answer[answer.find(":")+1:].strip()
            pairs.append({"question": question, "answer": answer})
    return pairs

# Generate 50 Q&A pairs
qa_pairs = generate_hr_qa_pairs(50)

# Save to JSON file
with open("qa_dataset.json", "w") as f:
    json.dump(qa_pairs, f, indent=2)

print(f"Generated {len(qa_pairs)} Q&A pairs and saved to qa_dataset.json")