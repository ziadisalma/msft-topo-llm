import os
import gc
import re
import random
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Constants and System Prompt ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
torch_dtype = torch.bfloat16
OPERATORS = ['+', '-', '*', '//']
OPERAND_RANGE = (1, 100)
SYSTEM_PROMPT = "You are a calculator. Evaluate the given mathematical expression and provide only the final numerical answer."

# --- Model and Tokenizer Loading ---
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    device_map="auto",
    attn_implementation="eager", # Use "flash_attention_2" for faster inference if available
)

# --- Model Configuration ---
model.config.output_attentions = True
model.config.output_hidden_states = True
model.config.use_cache = True
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
print("Model and tokenizer loaded successfully.")

# --- Helper Functions for Model Interaction and Analysis ---

def denormalize_token_list(tokenizer, words):
    """Converts a list of strings to their raw token representation (e.g., 'dog' -> 'Ġdog')."""
    denormalized_tokens = []
    for word in words:
        # Tokenize with a preceding space to get the model's internal token form.
        # We take the first result as words can sometimes be split.
        tokens = tokenizer.tokenize(f" {word}")
        if tokens:
            denormalized_tokens.append(tokens[0])
    return denormalized_tokens

def generate_with_outputs(system_prompt, user_prompt, max_new_tokens=128, temperature=0.1, top_p=0.9):
    """Generates text and returns the response, attentions, and embeddings."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt_with_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_with_template, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    # Generate output IDs using the model
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )[0]

    # Re-run the full sequence to get attentions and hidden states
    # This is necessary because generation mode with use_cache=True doesn't return them
    prev_cache_setting = model.config.use_cache
    model.config.use_cache = False
    with torch.no_grad():
        outputs = model(output_ids.unsqueeze(0), return_dict=True)
    model.config.use_cache = prev_cache_setting

    response_ids = output_ids[prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    attentions = outputs.attentions
    embeddings = outputs.hidden_states

    # Cleanup to free GPU memory
    del outputs, response_ids, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return response_text, attentions, embeddings, output_ids

def process_attentions(attentions, output_ids, tokenizer, target_tokens, query_position=-2, layers=None, head_average=True, as_numpy=True):
    """Extracts attention scores from a query token to a set of key tokens."""
    seq_tokens = tokenizer.convert_ids_to_tokens(output_ids)
    key_positions = [i for i, t in enumerate(seq_tokens) if t in target_tokens]
    if not key_positions: return {}

    # Default to analyzing all layers if none are specified
    total_layers = len(attentions)
    layer_indices = layers if layers is not None else list(range(1, total_layers + 1))
    
    processed_attns = {}
    for l in layer_indices:
        if l < 1 or l > total_layers: continue

        layer_attn = attentions[l-1][0] # Use l-1 for 0-based tensor index
        vec = layer_attn[:, query_position, key_positions] # (heads, num_key_positions)
        if head_average:
            vec = vec.mean(dim=0)
        if as_numpy:
            vec = vec.to(torch.float32).cpu().numpy()

        processed_attns[l] = {
            'query_token': seq_tokens[query_position],
            'key_tokens': [seq_tokens[i] for i in key_positions],
            'scores': vec
        }
    return processed_attns

# --- Experiment-Specific Functions ---

def generate_expression_and_answer(num_operands):
    """Recursively generates a mathematical expression and its correct answer."""
    if num_operands == 1:
        operand = random.randint(*OPERAND_RANGE)
        return str(operand), operand

    split_point = random.randint(1, num_operands - 1)
    left_expr, left_val = generate_expression_and_answer(split_point)
    right_expr, right_val = generate_expression_and_answer(num_operands - split_point)

    op = random.choice(OPERATORS)
    # Avoid division by zero
    if op == '//' and right_val == 0:
        op = random.choice(['+', '-', '*'])

    # Pad every token with spaces for cleaner tokenization
    expression = f"( {left_expr} {op} {right_expr} )"

    # Compute the numeric value
    value = eval(f"{left_val} {op} {right_val}")

    return expression, value


def run_arithmetic_experiment(ms, samples_per_m, layers_to_analyze=None):
    """
    Runs arithmetic experiments for varied numbers of operands (m),
    collecting model outputs and attention scores.
    """
    rows = []
    for m in ms:
        for i in range(samples_per_m):
            print(f"Running experiment for m={m}, sample {i+1}/{samples_per_m}...")
            expr, ans = generate_expression_and_answer(m)
            user_prompt = f"What is the value of the following expression?\n{expr}"

            # Get model outputs
            response, attentions, _, output_ids = generate_with_outputs(
                SYSTEM_PROMPT,
                user_prompt
            )

            # Find all numbers in the original expression to analyze attention
            operands_str = re.findall(r'\d+', expr)
            target_tokens = denormalize_token_list(tokenizer, operands_str)

            # Process attention from the token before last (likely the answer) to the operands
            # query_position=-2 is used because the final token is often <|eot_id|>
            attention_data = process_attentions(
                attentions,
                output_ids,
                tokenizer,
                target_tokens=target_tokens,
                query_position=-2, # The token before the end-of-sequence token
                layers=layers_to_analyze,
            )

            rows.append({
                'num_operands': m,
                'expression': expr,
                'prompt': user_prompt,
                'true_answer': ans,
                'model_answer': response.strip(),
                'attention_to_operands': attention_data,
            })

    return pd.DataFrame(rows)

# --- Main Execution Block ---

if __name__ == '__main__':
    # On non-CUDA systems, convert model to float32 for compatibility
    if not torch.cuda.is_available():
        model.to(torch.float32)

    # --- Experiment Parameters ---
    N_OPERANDS_TO_TEST = list(range(2, 6)) # Test expressions with 2, 3, 4, and 5 operands
    SAMPLES_PER_M = 3                    # Number of samples for each operand count
    # Specify which layers' attentions to analyze (e.g., early, middle, late)
    LAYERS_TO_ANALYZE = [1, 10, 20, 32]

    # --- Run Experiment ---
    results_df = run_arithmetic_experiment(
        ms=N_OPERANDS_TO_TEST,
        samples_per_m=SAMPLES_PER_M,
        layers_to_analyze=LAYERS_TO_ANALYZE
    )

    # --- Display Results ---
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    print("\n--- Experiment Results ---")
    print(results_df[['num_operands', 'expression', 'true_answer', 'model_answer', 'attention_to_operands']])
