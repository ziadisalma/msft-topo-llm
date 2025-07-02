import sys
import random
from prompting.wei_datasets import load_datasets, evaluate

def main():
    random.seed(0)
    gsm8k, svamp = load_datasets()
    datasets = [("GSM8K", gsm8k), ("SVAMP", svamp)]
    for name, ds in datasets:
        # Standard prompting evaluation
        corr_std, tot = evaluate(ds, k_shot=5, build_prompt_fn=build_standard_prompt, max_samples=100)
        print(f"{name} Standard Prompting: {corr_std}/{tot} = {corr_std/tot:.2%}")
        # Chain-of-thought prompting evaluation
        corr_cot, tot = evaluate(ds, k_shot=5, build_prompt_fn=build_cot_prompt, max_samples=100)
        print(f"{name} Chain-of-Thought Prompting: {corr_cot}/{tot} = {corr_cot/tot:.2%}\n")
