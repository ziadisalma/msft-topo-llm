import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Assuming these are your custom modules/functions
from arithmetic import generate_expression_and_answer, generate_reasoning_steps
from models.load_llama import extract_token_embeddings

# Configuration
M_VALUES = [2, 3, 4, 5, 6]
SAMPLES_PER_M = 10
K_EXAMPLES = 3
LAYERS = [4, 8, 12]
PROMPT_STYLES = ['zero_shot', 'few_shot', 'chain_of_thought']
OUTPUT_DIR = 'results'

# --- Start of Execution ---
print("--- Starting Intrinsic Dimension Experiment ---")
print(f"Configuration: M_VALUES={M_VALUES}, SAMPLES_PER_M={SAMPLES_PER_M}, K_EXAMPLES={K_EXAMPLES}, LAYERS={LAYERS}")
print(f"Output directory: '{OUTPUT_DIR}'")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_prompt_zero_shot(expr):
    return f"What is the value of the following expression?\n{expr}"

def build_prompt_few_shot(m):
    # sample K_EXAMPLES examples
    examples = []
    for _ in range(K_EXAMPLES):
        e_m = random.randint(2, m)
        e_expr, e_ans = generate_expression_and_answer(e_m)
        examples.append(f"{e_expr} = {e_ans}")
    main_expr, main_ans = generate_expression_and_answer(m)
    prompt = "Based on the examples below, solve the final expression.\n\n"
    prompt += "\n".join(examples)
    prompt += f"\n\nNow, solve this:\n{main_expr}"
    return prompt

def build_prompt_chain_of_thought(m):
    examples = []
    for i in range(K_EXAMPLES):
        e_m = random.randint(2, m)
        e_expr, e_ans = generate_expression_and_answer(e_m)
        reasoning = generate_reasoning_steps(e_expr)
        ex = (
            f"Example {i+1}:\n"
            f"Q: What is {e_expr}?\n"
            f"A: Let's solve it step-by-step:\n{reasoning}\nThe final answer is {e_ans}."
        )
        examples.append(ex)
    main_expr, main_ans = generate_expression_and_answer(m)
    prompt = (
        "Here are some examples with detailed reasoning. Use them to solve the final problem.\n\n"
        + "\n\n---\n\n".join(examples)
        + f"\n\n---\n\nNow, it's your turn:\nQ: What is {main_expr}?"
    )
    return prompt

# Helper: extract and pool embeddings for a batch of prompts
def get_pooled_embeddings(prompts, layer):
    # --- Print statement added ---
    print(f"    Extracting embeddings for {len(prompts)} prompts from layer {layer}...")
    pooled = []
    for i, text in enumerate(prompts):
        # --- Print statement added to show progress for long jobs ---
        if (i + 1) % 5 == 0: # Print every 5 prompts
             print(f"      Processing prompt {i+1}/{len(prompts)}...")
        emb_dict = extract_token_embeddings(text, layers=[layer])
        arr = emb_dict[layer]['embeddings']  # shape (num_tokens, hidden_size)
        pooled.append(arr.mean(axis=0))      # mean pooling
    return np.vstack(pooled)  # shape (num_prompts, hidden_size)

# Main loop: collect dims
scaling = {style: {layer: [] for layer in LAYERS} for style in PROMPT_STYLES}
for m in M_VALUES:
    # --- Print statement added ---
    print(f"\n{'='*20}\nProcessing for m={m} operands...\n{'='*20}")
    
    # build prompt lists
    prompts_by_style = {'zero_shot': [], 'few_shot': [], 'chain_of_thought': []}
    
    # --- Print statement added ---
    print(f"Generating {SAMPLES_PER_M} prompts for each style...")
    for i in range(SAMPLES_PER_M):
        expr, _ = generate_expression_and_answer(m)
        prompts_by_style['zero_shot'].append(build_prompt_zero_shot(expr))
        prompts_by_style['few_shot'].append(build_prompt_few_shot(m))
        prompts_by_style['chain_of_thought'].append(build_prompt_chain_of_thought(m))
    
    # for each style and layer, compute intrinsic dim
    for style in PROMPT_STYLES:
        # --- Print statement added ---
        print(f"\n  Analyzing style: '{style}'")
        for layer in LAYERS:
            # --- Print statement added ---
            print(f"  - Analyzing layer: {layer}")
            
            X = get_pooled_embeddings(prompts_by_style[style], layer)
            # --- Print statement added ---
            print(f"    Got pooled embeddings of shape: {X.shape}")
            
            pca = PCA()
            pca.fit(X)
            lambdas = pca.explained_variance_
            
            # participation ratio
            pr = (lambdas.sum()**2) / (np.sum(lambdas**2))
            # 90% variance dimension
            cum = np.cumsum(lambdas) / np.sum(lambdas)
            d90 = np.searchsorted(cum, 0.9) + 1
            
            scaling[style][layer].append((m, pr, d90))
            # --- Print statement added ---
            print(f"    Calculated Intrinsic Dimension: PR = {pr:.2f}, D90 = {d90}")


    # optionally: save eigenspectrum for last style/layer
    # --- Print statements added ---
    print("\n  Saving example eigenspectrum plot...")
    example_style = 'zero_shot'
    example_layer = LAYERS[-1]
    
    X_example = get_pooled_embeddings(prompts_by_style[example_style], example_layer)
    pca_ex = PCA()
    pca_ex.fit(X_example)
    plt.figure()
    plt.plot(pca_ex.explained_variance_, marker='o')
    plt.xlabel('Component index')
    plt.ylabel('Eigenvalue')
    plt.title(f'Eigenspectrum (m={m}, style={example_style}, layer={example_layer})')
    plt.tight_layout()
    
    filename = os.path.join(OUTPUT_DIR, f'eigenspectrum_m{m}_{example_style}_layer{example_layer}.pdf')
    plt.savefig(filename)
    plt.close()
    # --- Print statement added ---
    print(f"  Saved plot to '{filename}'")

# --- Print statement added ---
print(f"\n{'='*20}\nGenerating and saving final plots...\n{'='*20}")

# Plot scaling curves
for style in PROMPT_STYLES:
    # --- Print statement added ---
    print(f"Creating scaling plot for style: '{style}'")
    plt.figure()
    for layer in LAYERS:
        data = np.array(scaling[style][layer])  # columns: m, pr, d90
        ms = data[:,0]
        prs = data[:,1]
        plt.plot(ms, prs, marker='o', label=f'layer {layer}')
    plt.xlabel('Number of operands (m)')
    plt.ylabel('Participation ratio')
    plt.title(f'Intrinsic dimension scaling ({style})')
    plt.legend()
    plt.tight_layout()
    
    filename = os.path.join(OUTPUT_DIR, f'scaling_{style}.pdf')
    plt.savefig(filename)
    plt.close()
    # --- Print statement added ---
    print(f"  Saved plot to '{filename}'")


print("\nDone. PDFs saved in 'results/' directory.")
