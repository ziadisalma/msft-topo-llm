import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    output_attentions=True,
    attn_implementation="eager",  # ✅ Important for attention support
)

model.eval()

# Input prompt
prompt = "Given three lists (A, B, C) where only the largest element in each list can be moved to another list, and an element can only be added to a list if it is larger than all other elements in the list, determine if all elements of list A can be moved to list C in 7 moves? Only answer 'Yes' or 'No'. The lists are: A = [0, 1, 2], B = [] and C = []."
inputs = tokenizer(prompt, return_tensors="pt")

# Forward pass with attentions enabled explicitly
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions  # List: num_layers × (batch, num_heads, tgt_len, src_len)

# Quick check
assert attentions is not None, "Attention outputs are missing. Make sure output_attentions=True is passed to model()."

# Token strings and positions
token_strs = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
target_tokens = ["A", "B", "C", "0", "1", "2"]

# Find token indices
target_indices = [i for i, tok in enumerate(token_strs) if tok.strip("Ġ") in target_tokens]
# yes_index = next(i for i, tok in enumerate(token_strs) if "yes" in tok)
yes_index = next(
    i for i, tok in enumerate(token_strs) if tok.lstrip("Ġ▁") == "Yes"
)

if yes_index is None:
    raise ValueError("Token 'Yes' not found in the input token sequence.")

# Extract attention: from "yes" to each target token across layers
num_layers = len(attentions)
attention_from_yes = torch.zeros((len(target_indices), num_layers))

for layer_idx, layer_attn in enumerate(attentions):
    avg_heads = layer_attn[0].mean(dim=0)  # shape: (tgt_len, src_len)
    for i, idx in enumerate(target_indices):
        attention_from_yes[i, layer_idx] = avg_heads[yes_index, idx]

# Plot heat map
plt.figure(figsize=(10, 6))
sns.heatmap(attention_from_yes.numpy(), annot=False, cmap="plasma",
            xticklabels=[f"Layer {i}" for i in range(num_layers)],
            yticklabels=[token_strs[i] for i in target_indices])
plt.title('Attention from "Yes" to Tokens Across Layers')
plt.xlabel('Layers')
plt.ylabel('Token')
plt.tight_layout()
# plt.show()
plt.savefig("attention_heatmap.pdf")