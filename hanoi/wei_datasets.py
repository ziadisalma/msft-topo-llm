import pandas as pd
from datasets import load_dataset
import sys
import re
from pathlib import Path
import os

# temporary fix; will change later
try:
    repo_root = Path(__file__).resolve().parents[1]
except NameError:
    repo_root = Path(os.getcwd())

if repo_root.name == "prompting":
    repo_root = repo_root.parent

sys.path.append(str(repo_root))

from load_llama import generate_response

NUM_TEST_SAMPLES = 10
NUM_FEW_SHOT_EXEMPLARS = 8 
MAX_TOKENS_RESPONSE = 256
RESULTS_FILE = "wei_datasets_results.csv"

def extract_final_answer(text):
    match = re.search(r"####\s*([\d\.\,]+)", text)
    if match:
        return match.group(1).replace(",", "")
    return text.split("\n")[-1]

def build_prompt(few_shot_examples, test_question):
    prompt_text = ""
    for ex in few_shot_examples:
        prompt_text += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
    prompt_text = prompt_text.strip()
    prompt_text += f"\n\nQ: {test_question}\nA:"
    return prompt_text

def main():
    gsm8k_dataset = load_dataset("openai/gsm8k", "main")
    gsm8k_train = list(gsm8k_dataset['train'])
    gsm8k_test = list(gsm8k_dataset['test'])[:NUM_TEST_SAMPLES]

    svamp_dataset = load_dataset("ChilleD/svamp")
    svamp_train = list(svamp_dataset['train'])
    svamp_test = list(svamp_dataset['test'])[:NUM_TEST_SAMPLES]

    cot_exemplars = [
        {"question": item['question'], "answer": item['answer']}
        for item in gsm8k_train[:NUM_FEW_SHOT_EXEMPLARS]
    ]

    gsm8k_std_exemplars = [
        {"question": item['question'], "answer": extract_final_answer(item['answer'])}
        for item in gsm8k_train[:NUM_FEW_SHOT_EXEMPLARS]
    ]

    svamp_std_exemplars = [
        {"question": item['question_concat'], "answer": str(item['Answer'])}
        for item in svamp_train[:NUM_FEW_SHOT_EXEMPLARS]
    ]

    results = []

    for i, item in enumerate(gsm8k_test):
        test_question = item['question']
        ground_truth = item['answer']

        cot_prompt = build_prompt(cot_exemplars, test_question)
        cot_response = generate_response(cot_prompt, MAX_TOKENS_RESPONSE)
        results.append({
            "dataset": "GSM8K",
            "prompt_type": "Chain-of-Thought",
            "prompt": cot_prompt,
            "response": cot_response,
            "ground_truth": ground_truth
        })

        std_prompt = build_prompt(gsm8k_std_exemplars, test_question)
        std_response = generate_response(std_prompt, MAX_TOKENS_RESPONSE)
        results.append({
            "dataset": "GSM8K",
            "prompt_type": "Standard",
            "prompt": std_prompt,
            "response": std_response,
            "ground_truth": ground_truth
        })

    for i, item in enumerate(svamp_test):
        test_question = item['question_concat']
        ground_truth = str(item['Answer'])

        cot_prompt = build_prompt(cot_exemplars, test_question)
        cot_response = generate_response(cot_prompt, MAX_TOKENS_RESPONSE)
        results.append({
            "dataset": "SVAMP",
            "prompt_type": "Chain-of-Thought",
            "prompt": cot_prompt,
            "response": cot_response,
            "ground_truth": ground_truth
        })

        std_prompt = build_prompt(svamp_std_exemplars, test_question)
        std_response = generate_response(std_prompt, MAX_TOKENS_RESPONSE)
        results.append({
            "dataset": "SVAMP",
            "prompt_type": "Standard",
            "prompt": std_prompt,
            "response": std_response,
            "ground_truth": ground_truth
        })

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)
    print("Script finished successfully.")

if __name__ == "__main__":
    main()
