import random
from datasets import load_dataset
from load_llama import generate_response

def load_datasets():
    gsm8k = load_dataset("gsm8k", "main", split="test")
    svamp = load_dataset("svamp", split="test")
    return gsm8k, svamp

def sample_examples(dataset, k):
    return random.sample(list(dataset), k)

def build_standard_prompt(examples, question):
    prompt = ""
    for ex in examples:
        prompt += f"Question: {ex['question'].strip()}\nAnswer: {ex['answer'].strip()}\n\n"
    prompt += f"Question: {question.strip()}\nAnswer:"
    return prompt

def build_cot_prompt(examples, question):
    prompt = ""
    for ex in examples:
        rationale = ex.get('solution', '').strip()
        prompt += f"Question: {ex['question'].strip()}\nLet's think step by step.\n{rationale}\nAnswer: {ex['answer'].strip()}\n\n"
    prompt += f"Question: {question.strip()}\nLet's think step by step.\n"
    return prompt

def extract_answer(response):
    return response.strip().split()[-1]

def evaluate(dataset, k_shot, build_prompt_fn, max_samples=None):
    data = list(dataset)
    if max_samples:
        data = data[:max_samples]
    total = len(data)
    correct = 0
    for ex in data:
        examples = sample_examples(dataset, k_shot)
        prompt = build_prompt_fn(examples, ex['question'])
        response = generate_response(prompt, max_new_tokens=64)
        pred = extract_answer(response)
        if pred == ex['answer'].strip():
            correct += 1
    return correct, total
