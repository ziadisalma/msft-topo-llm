import pandas as pd
from datasets import load_dataset
import sys
import re
from pathlib import Path
import os

# --- Boilerplate to add the repository root to the path ---
try:
    # Handles running from within the 'prompting' directory or the repo root
    repo_root = Path(__file__).resolve().parent
    if repo_root.name == "prompting":
        repo_root = repo_root.parent
    else:
        # Assumes script is run from the repo root
        pass
except NameError:
    # Fallback for interactive environments
    repo_root = Path(os.getcwd())

sys.path.append(str(repo_root))
# -----------------------------------------------------------

from load_llama import generate_response

# --- Configuration ---
NUM_TEST_SAMPLES = 5  # Number of samples to test from each dataset
NUM_FEW_SHOT_EXEMPLARS = 2 # Using a smaller subset for faster execution
MAX_TOKENS_RESPONSE = 256 # Max tokens for the model's response
RESULTS_FILE = "wei_datasets_results_revised.csv"

def extract_final_answer(text):
    """Extracts the final numerical answer from a GSM8K-style CoT string."""
    # Searches for the answer after the '####' delimiter.
    match = re.search(r"####\s*([\d\.\,]+)", text)
    if match:
        return match.group(1).replace(",", "")
    
    # Fallback for simple answers or malformed CoT strings
    parts = text.split("\n")
    return parts[-1].strip() if parts[-1].strip() else parts[-2].strip()

def build_standard_prompt(test_question):
    """Builds a zero-shot prompt with only the question."""
    return f"Q: {test_question}\nA:"

def build_cot_prompt(few_shot_examples, test_question):
    """Builds a few-shot prompt with an explicit instruction to think step-by-step."""
    # Instruction for the model, followed by examples
    prompt_text = "Let's think step by step to solve the following math problems.\n\n"
    
    for ex in few_shot_examples:
        prompt_text += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
    
    # Remove the final newlines to properly place the test question
    prompt_text = prompt_text.strip()
    prompt_text += f"\n\nQ: {test_question}\nA:"
    return prompt_text

def main():
    """
    Main function to load datasets, generate prompts, run the model,
    and save the results.
    """
    print("Loading datasets...")
    # Load GSM8K dataset from the original source
    gsm8k_dataset = load_dataset("openai/gsm8k", "main")
    gsm8k_train = list(gsm8k_dataset['train'])
    gsm8k_test = list(gsm8k_dataset['test'])[:NUM_TEST_SAMPLES]

    # Load SVAMP dataset
    svamp_dataset = load_dataset("ChilleD/svamp")
    svamp_test = list(svamp_dataset['test'])[:NUM_TEST_SAMPLES]
    print("Datasets loaded.")

    print(f"Preparing {NUM_FEW_SHOT_EXEMPLARS} few-shot exemplars for CoT...")
    # CoT exemplars from GSM8K are used for both datasets to replicate the paper's strategy
    cot_exemplars = [
        {"question": item['question'], "answer": item['answer']}
        for item in gsm8k_train[:NUM_FEW_SHOT_EXEMPLARS]
    ]
    print("Exemplars prepared.")

    results = []

    # --- Process GSM8K ---
    print("\n--- Testing on GSM8K ---")
    for i, item in enumerate(gsm8k_test):
        print(f"\nProcessing GSM8K sample {i+1}/{len(gsm8k_test)}...")
        test_question = item['question']
        ground_truth = item['answer']

        # 1. Chain-of-Thought Prompting
        cot_prompt = build_cot_prompt(cot_exemplars, test_question)
        cot_response = generate_response(cot_prompt, MAX_TOKENS_RESPONSE)
        results.append({
            "dataset": "GSM8K",
            "prompt_type": "Chain-of-Thought",
            "prompt": cot_prompt,
            "response": cot_response,
            "ground_truth": ground_truth
        })

        # 2. Standard (Zero-Shot) Prompting
        std_prompt = build_standard_prompt(test_question)
        std_response = generate_response(std_prompt, MAX_TOKENS_RESPONSE)
        results.append({
            "dataset": "GSM8K",
            "prompt_type": "Standard",
            "prompt": std_prompt,
            "response": std_response,
            "ground_truth": ground_truth
        })

    # --- Process SVAMP ---
    print("\n--- Testing on SVAMP ---")
    for i, item in enumerate(svamp_test):
        print(f"\nProcessing SVAMP sample {i+1}/{len(svamp_test)}...")
        test_question = item['Body'] + " " + item["Question"] # Use Body and Question for full context
        ground_truth = str(item['Answer'])

        # 1. Chain-of-Thought Prompting (using GSM8K exemplars)
        cot_prompt = build_cot_prompt(cot_exemplars, test_question)
        cot_response = generate_response(cot_prompt, MAX_TOKENS_RESPONSE)
        results.append({
            "dataset": "SVAMP",
            "prompt_type": "Chain-of-Thought",
            "prompt": cot_prompt,
            "response": cot_response,
            "ground_truth": ground_truth
        })

        # 2. Standard (Zero-Shot) Prompting
        std_prompt = build_standard_prompt(test_question)
        std_response = generate_response(std_prompt, MAX_TOKENS_RESPONSE)
        results.append({
            "dataset": "SVAMP",
            "prompt_type": "Standard",
            "prompt": std_prompt,
            "response": std_response,
            "ground_truth": ground_truth
        })

    # --- Save Results ---
    print(f"\nSaving results to {RESULTS_FILE}...")
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)
    print("Script finished successfully.")

if __name__ == "__main__":
    main()
