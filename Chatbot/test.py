from fastapi import FastAPI
from med_assist.chain import build_chain
from dotenv import load_dotenv
from med_assist.components.embeddings import check_for_embeddings, create_embeddings
import json
import asyncio

if not check_for_embeddings():
    create_embeddings()

chain = build_chain()

# Load questions from questions.jsonl
with open('questions.jsonl', 'r') as f:
    questions_data = json.load(f)

async def process_questions():
    results = {}
    for level, questions in questions_data.items():
        results[level] = []
        for question in questions:
            answer = await chain.ainvoke(question)
            results[level].append({"question": question, "answer": answer})
    
    # Save results to a new file
    with open('answers.json', 'w') as f:
        json.dump(results, f, indent=2)

# Run the async function
asyncio.run(process_questions())


if __name__ == "__main__":
    print("Questions processed and answers saved to 'answers.json'")


