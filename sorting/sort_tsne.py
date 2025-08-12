import os
import re
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from models.load_llama import tokenizer, extract_token_embeddings

OUTPUT_PATH = "rsa_tsne_sort_using_lib.pdf"
NUMBERS = [1, 7, 4, 6, 2, 5]
K = 3
Kth = "third"

def main():
    system_prompt = (
        "You are an expert in numerical reasoning. "
        "Respond with only the single number that is the k-th smallest."
    )
    user_prompt = (
        f"Here is a list of numbers: {', '.join(map(str, NUMBERS))}.\n"
        f"Find the {Kth} smallest number."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Define token strings to match numeric values
    num_strs = [str(n) for n in NUMBERS]
    token_filters = num_strs + [f"Ġ{n}" for n in NUMBERS]

    # Extract embeddings for numeric tokens across all layers
    emb_dict = extract_token_embeddings(
        text=prompt,
        tokens=token_filters,
        layers=None,
        as_numpy=True
    )

    # Collect embeddings in layer order
    layer_indices = sorted(emb_dict.keys())  # 1..L
    all_emb = [emb_dict[l]['embeddings'] for l in layer_indices]  # each is (T, D)
    all_emb = np.stack(all_emb, axis=0)  # shape: (L, T, D)

    L, T, D = all_emb.shape
    # Flatten and normalize for t-SNE
    flat_emb = all_emb.reshape(L * T, D)
    flat_emb = flat_emb / np.linalg.norm(flat_emb, axis=1, keepdims=True)

    # Compute ranks and values
    sorted_nums = sorted(NUMBERS)
    ranks = [sorted_nums.index(v) + 1 for v in NUMBERS]
    # Expand across layers
    rank_ids = np.tile(np.array(ranks), L)
    value_ids = np.tile(np.array(NUMBERS), L)

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, metric='cosine', random_state=42)
    proj = tsne.fit_transform(flat_emb)  # → (L*T, 2)
    proj = proj.reshape(L, T, 2)

    # Plot trajectories
    plt.figure(figsize=(6,6))
    cmap = plt.get_cmap("tab10")
    k_val = sorted_nums[K-1]

    for t in range(T):
        xs, ys = proj[:, t, 0], proj[:, t, 1]
        rank = rank_ids[t]
        val = value_ids[t]
        color = cmap((rank-1) % 10)
        lw = 3 if val == k_val else 1
        alpha = 1.0 if val == k_val else 0.4

        plt.plot(xs, ys, c=color, linewidth=lw, alpha=alpha)
        plt.scatter(xs, ys, c=[color], s=50 if val==k_val else 30, alpha=alpha)
        for layer_idx, (x, y) in enumerate(zip(xs, ys), start=1):
            plt.text(
                x, y, str(layer_idx),
                fontsize=6,
                fontweight='bold' if (val==k_val and layer_idx==L) else 'normal',
                ha='center', va='center',
                color='white' if val==k_val else 'black',
                alpha=alpha
            )

    # Annotate start and highlight end
    for t in range(T):
        val = value_ids[t]
        plt.text(proj[0, t, 0], proj[0, t, 1], f"{val}", fontsize=10)
        if val == k_val:
            plt.text(
                proj[-1, t, 0], proj[-1, t, 1], f"{val}",
                fontsize=14, fontweight="bold"
            )

    plt.title(f"t-SNE trajectories {Kth}-smallest using load_llama lib")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"Saved t-SNE trajectory plot to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
