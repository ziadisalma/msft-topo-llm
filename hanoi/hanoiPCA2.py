import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# choose model :)
models = ["meta-llama/Meta-Llama-3.1-8B-Instruct", "microsoft/phi-4", "Qwen/Qwen2.5-7B-Instruct-1M"]
model_name =  models[2]
# focus_tokens = ["A", "B", "C", "0", "1", "2"]
focus_tokens = ["solve", "ascending", "Step", "steps", "minimum"]

which_model = "llama"
if model_name == models[1]:
    which_model = "phi-4"

# choose prompt :3
"""
prompt = (
    "Given three lists, A, B, and C, where only the largest element in each list "
    "can be moved to another list, and an element can only be added to a list if "
    "it is larger than all other elements in the list, determine if all elements of list A can be moved to list C."
    "The lists are: A = [0, 1, 2], B = [], C = []."
)
"""

prompts = [""" Problem description:
- There are three lists labeled A, B, and C.
- There is a set of numbers distributed among those three lists.
- You can only move numbers from the rightmost end of one list to
the rightmost end of another list.
Rule #1:  You can only move a number if it is at the rightmost end
of its current list.
Rule #2:  You can only move a number to the rightmost end of a
list if it is larger than the other numbers in that list.
A move is valid if it satisfies both Rule #1 and Rule #2.
A move is invalid if it violates either Rule #1 or Rule #2.

Goal:  The goal is to end up in the configuration where all
numbers are in list C, in ascending order using minimum number
of moves.

Here are two examples:
Example 1:

This is the starting configuration:
A = [0, 1]
B = [2]
C = []

This is the goal configuration:
A = []
B = []
C = [0, 1, 2]

Here is the sequence of minimum number of moves to reach the goal
configuration from the starting configuration:

Move 2 from B to C.
A = [0, 1]
B = []
C = [2]

Move 1 from A to B.
A = [0]
B = [1]
C = [2]

Move 2 from C to B.
A = [0]
B = [1, 2]
C = []

Move 0 from A to C.
A = []
B = [1, 2]
C = [0]

Move 2 from B to A.
A = [2]
B = [1]
C = [0]

Move 1 from B to C.
A = [2]
B = []
C = [0, 1]

Move 2 from A to C.
A = []
B = []
C = [0, 1, 2]

Example 2:
This is the starting configuration:
A = [1]
B = [0]
C = [2]

This is the goal configuration:
A = []
B = []
C = [0, 1, 2]

Here is the sequence of minimum number of moves to reach the goal
configuration from the starting configuration:

Move 2 from C to A.
A = [1, 2]
B = [0]
C = []

Move 0 from B to C.
A = [1, 2]
B = []
C = [0]

Move 2 from A to B.
A = [1]
B = [2]
C = [0]

Move 1 from A to C.
A = []
B = [2]
C = [0, 1]

Move 2 from B to C.
A = []
B = []
C = [0, 1, 2]

This is the starting configuration:
A = [0, 1, 2]
B = []
C = []

This is the goal configuration:
A = []
B = []
C = [0,1,2]

Give me the sequence of moves to solve the puzzle from the
starting configuration, updating the lists after each move.
Please try to use as few moves as possible, and make sure to
follow Rule #1 and Rule #2.  Please limit your answer to a
maximum of 10 steps.

Please format your answer as below:
Step 1.  Move <N> from <src> to <tgt>.
A = []
B = []
C = []
""",

""" Problem description:
- There are three lists labeled A, B, and C.
- There is a set of numbers distributed among those three lists.
- You can only move numbers from the rightmost end of one list to
the rightmost end of another list.
Rule #1:  You can only move a number if it is at the rightmost end
of its current list.
Rule #2:  You can only move a number to the rightmost end of a
list if it is larger than the other numbers in that list.
A move is valid if it satisfies both Rule #1 and Rule #2.
A move is invalid if it violates either Rule #1 or Rule #2.

Goal:  The goal is to end up in the configuration where all
numbers are in list C, in ascending order using minimum number
of moves.

Give me the sequence of moves to solve the puzzle from the
starting configuration, updating the lists after each move.
Please try to use as few moves as possible, and make sure to
follow Rule #1 and Rule #2.  Please limit your answer to a
maximum of 10 steps.

Please format your answer as below:
Step 1.  Move <N> from <src> to <tgt>.
A = []
B = []
C = []
"""]

prompt = prompts[0]
title_prompt = "No-Shot"
if prompt == prompts[0]:
    title_prompt = "Few-Shot"

def extract_layer_activations(model, tokenizer, texts, focus_tokens=None, device='cpu'):
    """Extract hidden states for focus tokens from all layers."""
    model.to(device)
    model.eval()

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Find positions of focus tokens in tokenized prompt
    if focus_tokens:
        focus_indices = []
        for idx, token in enumerate(tokens):
            for focus in focus_tokens:
                if focus.lower() in token.lower():
                    focus_indices.append(idx)
        if not focus_indices:
            print("Warning: No focus tokens found in the input.")
    else:
        focus_indices = list(range(len(tokens)))  # fallback: use all

    # Extract activations at focus token positions
    hidden_states = outputs.hidden_states  # tuple: (layer0, layer1, ..., layerN)
    selected_activations = []
    for layer in hidden_states:
        layer_focus = layer[0, focus_indices, :]  # shape: (num_focus_tokens, hidden_size)
        selected_activations.append(layer_focus.cpu().numpy())
    
    return selected_activations  # List of (num_focus_tokens, hidden_size)

def compute_participation_ratio(layer_activations):
    """Compute participation ratio from PCA eigenvalues."""
    pca = PCA()
    pca.fit(layer_activations)
    eigenvalues = pca.explained_variance_
    
    pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    return pr

def plot_sorted_eigenvalues(layer_activations, which_model, title_prompt):
    for i, layer in enumerate(layer_activations):
        pca = PCA()
        pca.fit(layer)
        eigenvalues = pca.explained_variance_
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]

        plt.figure(figsize=(6, 4))
        plt.plot(sorted_eigenvalues, marker='o')
        plt.title(f"Layer {i} PCA Eigenvalues\n{which_model} - {title_prompt}")
        plt.xlabel("Component")
        plt.ylabel("Eigenvalue")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"eigenvalues_{which_model}_{title_prompt}_layer{i}.pdf")
        plt.close()

def plot_participation_ratios(prs):
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(prs)), prs, marker='o')
    plt.xlabel("Layer")
    plt.ylabel("Participation Ratio")
    plt.title("Participation Ratio vs Layer")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    focus = focus_tokens
    print(f"Using model: {model_name}, Prompt type: {title_prompt}, Focus tokens: {focus}")

    layer_activations = extract_layer_activations(model, tokenizer, prompt, focus_tokens=focus, device=device)

    # Plot PCA eigenvalues for each layer using focus tokens only
    plot_sorted_eigenvalues(layer_activations, which_model, title_prompt)
    plt.savefig(f"pca_eig_{which_model}_{title_prompt}.pdf")

if __name__ == "__main__":
    main()