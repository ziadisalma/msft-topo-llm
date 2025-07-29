import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# --- Configuration ---
models = ["meta-llama/Meta-Llama-3.1-8B-Instruct", "microsoft/phi-4", "Qwen/Qwen2.5-7B-Instruct-1M"]
model_name =  models[2]
focus_tokens = ["A", "B", "C", "0", "1", "2"] # , "3"]
# LAYERS_TO_PLOT = list(range(1, 33))  # Llama-3 has 32 layers
# LAYERS_TO_PLOT = list(range(1, 41))  # Phi-4 has 40 layers
LAYERS_TO_PLOT = list(range(1, 29))  # Qwen has 28 layers
SEED = 42
PERPLEXITY = 5

# --- Load model and tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto").eval()

# --- Input prompt ---
"""
prompt = (
    "Given three lists, A, B, and C, where only the largest element in each list "
    "can be moved to another list, and an element can only be added to a list if "
    "it is larger than all other elements in the list, determine if all elements of list A can be moved to list C."
    "The lists are: A = [0, 1, 2], B = [], C = []."
)
"""

prompt = """ Problem description:
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
"""

# --- Tokenize ---
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_ids = inputs["input_ids"]
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# --- Identify positions of target tokens in tokenized input ---
def find_token_positions(target_tokens, tokens):
    positions = {}
    for target in target_tokens:
        for idx, tok in enumerate(tokens):
            if target in tok and target not in positions:
                positions[target] = idx
                break
        if target not in positions:
            print(f"Warning: Token '{target}' not found in tokenized prompt.")
    return positions


token_positions = find_token_positions(focus_tokens, tokens)

# --- Get hidden states across layers ---
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple of (layer+1) tensors [batch, seq_len, hidden_dim]

# --- Collect embeddings per layer for each target token ---
layerwise_embeddings = {token: [] for token in focus_tokens}

for layer in LAYERS_TO_PLOT:
    layer_embed = hidden_states[layer][0]  # [seq_len, hidden_dim]
    for token, pos in token_positions.items():
        emb = layer_embed[pos].cpu().numpy()
        layerwise_embeddings[token].append(emb)

# --- Flatten and prepare for t-SNE ---
all_embeddings = []
labels = []
for token, embs in layerwise_embeddings.items():
    for l, emb in enumerate(embs):
        all_embeddings.append(emb)
        labels.append(f"{token}-L{l+1}")

# --- Run t-SNE ---
tsne = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=SEED)
reduced = tsne.fit_transform(np.array(all_embeddings))

# --- Plot ---
plt.figure(figsize=(10, 8))
colors = {
    "A": "red",
    "B": "blue",
    "C": "green",
    "0": "brown",
    "1": "purple",
    "2": "orange" # ,
    #"3": "brown"
}

for token in focus_tokens:
    xs, ys = [], []
    for i, label in enumerate(labels):
        if label.startswith(token):
            xs.append(reduced[i][0])
            ys.append(reduced[i][1])
    plt.plot(xs, ys, marker='o', color=colors[token], label=token)
    for i in range(len(xs)):
        if i == len(xs) - 1:
            plt.text(xs[i] + 0.5, ys[i], f"{token}", fontsize=10, fontweight='bold')

plt.title("t-SNE of Token Embeddings Across Layers (QWEN2.5 7B)") # (PHI-4 14B)") # (LLaMA 3.1 8B)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig("octopus_plot1_Hanoi_qwen.pdf")