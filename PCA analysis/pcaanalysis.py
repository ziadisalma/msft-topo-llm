import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# choose model :)
models = ["meta-llama/Meta-Llama-3.1-8B-Instruct", "microsoft/phi-4", "Qwen/Qwen2.5-7B-Instruct-1M"]
model_name =  models[0]
focus_tokens_list = [None, ["node", "present", "last"], ["higher", "highest", "high", "greater", "great", "comparing", "compar", "compared"], ["node", "present", "last", "higher", "highest", "high", "greater", "great", "comparing", "compar", "compared"]]
focus_tokens = focus_tokens_list[1]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

which_model = "llama"
if model_name == models[1]:
    which_model = "phi-4"
if model_name == models[2]:
    which_model = "qwen"

# prompt :3
# prompt = ['<|endoftext|>Q: P: lobby-71-20-5-81-75. P-rewards=[75:25 vs 43:59]. S: lobby-83-66-5. S-rewards=[5:83 vs 11:79].\nA: S\n\nQ: P: lobby-44-100. P-rewards=[22:47 vs 100:75]. S: lobby-6-35. S-rewards=[35:38 vs 21:51].\nA: P\n\nQ: P: lobby-59-57-16-11-31. P-rewards=[58:10 vs 31:57]. S: lobby-98-20. S-rewards=[32:55 vs 20:79].\nA: S\n\nQ: P: lobby-98-54-6-34-66. P-rewards=[66:18 vs 29:18]. S: lobby-52-39-62-46-75. S-rewards=[74:97 vs 75:37].\nA: S\n\nQ: P: lobby-38-69-77. P-rewards=[3:24 vs 77:1]. S: lobby-62-50-78-76-30. S-rewards=[30:95 vs 94:39].\nA: S\n\nQ: P: lobby-86-93-85-29-34. P-rewards=[34:70 vs 51:94]. S: lobby-86-21-40-3. S-rewards=[3:8 vs 80:81].\nA:']

role = "You are given two paths and rewards for a node on the path."
prompt = '<|endoftext|>Q: path1: X-U-L-D. path1-rewards=[D:2 vs F:28]. path2: T-W-H-P-K. path2-rewards=[F:93 vs K:68].\nA: K\n\nQ: path1: A-U-W-C-H. path1-rewards=[H:32 vs L:100]. path2: F-J. path2-rewards=[J:6 vs L:51].\nA: H\n\nQ: path1: E-A-S-Z-D. path1-rewards=[C:80 vs D:59]. path2: R-I-G-F-U. path2-rewards=[C:31 vs U:12].\nA: D\n\nQ: path1: W-Y. path1-rewards=[Y:59 vs D:64]. path2: E-O-L-V. path2-rewards=[V:77 vs D:78].\nA: V\n\nQ: path1: X-I-Y-V-Z. path1-rewards=[H:96 vs Z:38]. path2: N-W-B-A-S. path2-rewards=[H:45 vs S:25].\nA: Z\n\nQ: path1: E-R. path1-rewards=[Z:7 vs R:94]. path2: A-P. path2-rewards=[P:50 vs Z:78].\nA: R\n\nQ: path1: T-A. path1-rewards=[A:21 vs U:1]. path2: C-Y. path2-rewards=[U:62 vs Y:57].\nA: Y\n\nQ: path1: K-L. path1-rewards=[G:77 vs L:2]. path2: D-O-C. path2-rewards=[G:10 vs C:67].\nA: C\n\nQ: path1: F-P. path1-rewards=[P:59 vs N:89]. path2: Q-O-K-C. path2-rewards=[C:57 vs N:35].\nA: P\n\nQ: path1: J-D-M. path1-rewards=[M:30 vs R:92]. path2: B-L-P. path2-rewards=[R:84 vs P:92].\nA: P\n\nQ: path1: T-G-I. path1-rewards=[I:40 vs B:34]. path2: O-Z-W-P. path2-rewards=[B:55 vs P:58].\nA:'

messages = [
    {"role": "system", "content": role},
    {"role": "user", "content": prompt},
]

title_prompt = "Gen1"
# if prompt == prompts[1]:
#     title_prompt = "Gen2"

import torch

def extract_layer_activations(gen_out, tokenizer, focus_tokens=None, device='cpu'):
    """
    Generate from `messages`, then extract per-layer activations at the last position
    for generated tokens that match any of `focus_tokens`.
    Returns: [num_layers+1 x (num_focus, hidden)], generated_tokens(list[str])
    """

    # --- Recover generated token IDs/tokens ---
    # scores: list of length gen_len, each (batch, vocab)
    scores = torch.stack(gen_out.scores, dim=1)  # (batch, gen_len, vocab)
    gen_ids = scores.argmax(dim=-1)[0]          # (gen_len,)
    generated_tokens = tokenizer.convert_ids_to_tokens(gen_ids.tolist())
     # strip prompt
    reply   = tokenizer.decode(gen_ids, skip_special_tokens=False)
    print(reply)
    # --- Choose focus steps (indices within generated tokens) ---
    if focus_tokens:
        focus_lower = [f.lower() for f in focus_tokens]
        focus_steps = [i for i, tok in enumerate(generated_tokens)
                       if any(f in tok.lower() for f in focus_lower)]
        if not focus_steps:
            print("Warning: No focus tokens found in the generated output.")
    else:
        focus_steps = list(range(len(generated_tokens)))

    # --- Collect last-position hidden states for each generated step & layer ---
    # hidden_states: list length gen_len;
    #   hidden_states[t] is a tuple (layer0..layerN), each (batch, seq_len, hidden)
    if len(gen_out.hidden_states) == 0:
        raise RuntimeError("No hidden states returned. Ensure output_hidden_states=True.")

    num_layers_plus_embed = len(gen_out.hidden_states[0])  # usually num_layers+1 (incl. embeddings)
    selected_activations = []
    for layer_idx in range(num_layers_plus_embed):
        vecs = []
        for step_idx in focus_steps:
            h = gen_out.hidden_states[step_idx][layer_idx]  # (batch, seq_len, hidden)
            vec = h[0, -1, :].detach().to(torch.float32).cpu().numpy()
            vecs.append(vec)    
        if len(vecs) == 0:
            selected_activations.append(np.empty((0, gen_out.hidden_states[0][layer_idx].shape[-1])))
        else:
            selected_activations.append(np.stack(vecs, axis=0))  # (num_focus, hidden)
    return selected_activations, generated_tokens

def compute_participation_ratio(layer_activations):
    """Compute participation ratio from PCA eigenvalues."""
    pca = PCA()
    pca.fit(layer_activations)
    eigenvalues = pca.explained_variance_
    
    explained_variance_ratio = pca.explained_variance_ratio_
    total_dim = len(explained_variance_ratio)
    
    pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    return pr, total_dim

def plot_sorted_eigenvalues(layer_activations, which_model, title_prompt):
    for i, layer in enumerate(layer_activations):
        pca = PCA()
        pca.fit(layer)
        eigenvalues = pca.explained_variance_
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]

        plt.figure(figsize=(6, 4))
        plt.plot(sorted_eigenvalues[:12], marker='o')
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

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = 'left'
    model.to(device)
    model.eval()

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    # Some chat models use <|eot_id|>; guard if it's missing
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        terminators = [tokenizer.eos_token_id, eot_id] if eot_id is not None else [tokenizer.eos_token_id]
    except Exception:
        terminators = [tokenizer.eos_token_id]


    max_new_tokens = 128
    with torch.no_grad():
        gen_out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )

    for k in range(3,4):
        print(f"Using model: {model_name}, Prompt type: {title_prompt}, Focus tokens: {focus_tokens_list[k]}")

        # Extract activations from all layers (full tokens)
        
        layer_activations_all, predicted_tokens = extract_layer_activations(
            gen_out, tokenizer, focus_tokens=focus_tokens_list[k], device=device
        )
        
        prs = []
        for i, layer in enumerate(layer_activations_all):
            if layer.ndim != 2:
               print(f"Skipping PCA on Layer {i} due to invalid shape: {layer.shape}")
               continue
            pr, total_dim = compute_participation_ratio(layer)
            prs.append(pr)
            #print(f"Layer {i}: Participation Ratio = {pr:.2f} for {which_model} on {title_prompt} prompt (total dim: {total_dim})")
            print(pr)
        # plot_participation_ratios(prs)
        # plt.savefig(f"prs_{which_model}_{title_prompt}_{k}.pdf")

    # Save participation ratio plot (optional)
    # plot_participation_ratios(prs)
    # plt.savefig(f"prs_{which_model}_{title_prompt}.pdf")

    # Extract activations again but only at focus tokens
    
    layer_activations_focus, _ = extract_layer_activations(
        gen_out, tokenizer, focus_tokens=focus_tokens_list[k], device=device
    )

    # Plot sorted PCA eigenvalues for each layer using focus tokens only
    # for i, layer in enumerate(layer_activations_focus):
    #     if layer.ndim != 2:
    #         print(f"Skipping eigenvalue plot for Layer {i} due to invalid shape: {layer.shape}")
    #         continue
    #     plot_participation_ratios(prs)
    #     plt.savefig(f"prs_{which_model}_{title_prompt}.pdf")
    #     plot_sorted_eigenvalues([layer], which_model, title_prompt)

if __name__ == "__main__":
    main()
