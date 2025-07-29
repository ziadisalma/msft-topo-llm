import numpy as np
import matplotlib.pyplot as plt
from models.load_llama import extract_attention_to_tokens

NUMBERS = [1, 7, 4, 6, 2, 5]
K = 3
SAVE_PATH = "sort_attn_heatmap.pdf"


def kth_suffix(k: int) -> str:
    if 11 <= (k % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")

def main():
    system_prompt = (
        "You are an expert in numerical reasoning. "
        "Respond with only the single number that is the k-th smallest."
    )
    user_prompt = (
        f"Here is a list of numbers: {', '.join(map(str, NUMBERS))}.\n"
        f"Find the {K}{kth_suffix(K)}-smallest number."
    )

    num_strs = [str(n) for n in NUMBERS]
    token_filters = num_strs + [f"Ġ{n}" for n in NUMBERS]

    attn_data = extract_attention_to_tokens(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tokens=token_filters,
        layers=None,
        max_new_tokens=1,
        head_average=True,
        as_numpy=True
    )

    answer = attn_data['answer']
    tokens = attn_data['tokens']
    layers = attn_data['layers']
    attentions = attn_data['attentions']

    print(f"Answer: {answer}")

    matrix = np.stack([attentions[l] for l in layers], axis=0)  # shape (L, T)

    L, T = matrix.shape
    plt.figure(figsize=(max(6, 0.7 * T), max(4, 0.3 * L)))
    im = plt.imshow(matrix, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar(im, label="Attention weight (answer → token)")

    plt.xticks(ticks=np.arange(T), labels=tokens, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(L), labels=layers)
    plt.xlabel("Token in prompt (unsorted list)")
    plt.ylabel("Layer")
    plt.title(
        f"Answer-token attention per layer to each list element\n"
        f"K = {K}{kth_suffix(K)}-smallest"
    )
    plt.tight_layout()

    if SAVE_PATH:
        plt.savefig(SAVE_PATH, dpi=300)
        print(f"Saved heatmap to {SAVE_PATH}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
