import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


model_name = "microsoft/Phi-4-reasoning"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, torch_dtype= torch.float16, device_map="auto")
model.eval()


prompt = (
    "Given the following undirected graph description, determine whether the graph would "
    "still be connected after removing the node Z and its incident edges: "
    " The graph has 6 nodes"
    "A is connected O which is connected to F and Z. Z is conncted to R which is connected to G."
    "Use minimal reasoning"
)


inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_ids = inputs["input_ids"][0] 
tokens = tokenizer.convert_ids_to_tokens(input_ids)
clean_tokens = [t.lstrip("Ġ") for t in tokens]



with torch.no_grad():
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states           
    gen_ids = model.generate(
	inputs["input_ids"],
	max_new_tokens = 4096,
	do_sample=True,
    temperature = 0.8,
    top_k=50,
    top_p=0.95,
    )
    generated = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    print(generated)

all_layers = torch.stack(hidden_states, dim=0)[:, 0, :, :]  
num_layers, seq_len, hidden_size = all_layers.shape


nodes = ["F","O", "Z", "R", "A", "G"]
node_indices = [i for i, tk in enumerate(clean_tokens) if tk in nodes]
node_labels = [clean_tokens[i] for i in node_indices]
n_nodes = len(node_indices)


selected = all_layers[:, node_indices, :]


flat = selected.reshape(num_layers * n_nodes, hidden_size).cpu().numpy()

tsne = TSNE(
    n_components=2,
    perplexity=30,     
    init="pca",
    random_state=42
)
flat_2d = tsne.fit_transform(flat)


traj = flat_2d.reshape(num_layers, n_nodes, 2)


plt.figure(figsize=(8, 6))
for j, lbl in enumerate(node_labels):
    x = traj[:, j, 0]
    y = traj[:, j, 1]
    plt.plot(x, y, marker="o", label=lbl)
    for layer in range(num_layers):
        plt.text(x[layer], y[layer], str(layer), fontsize=7, alpha=0.6)

plt.title("t‑SNE Trajectories of the Input Nodes Across Layers")
plt.xlabel("t‑SNE Dim 1")
plt.ylabel("t‑SNE Dim 2")
plt.legend(title="Node", bbox_to_anchor=(1.05,1), loc="upper left", fontsize="small")
plt.tight_layout()

# 11. Save to PDF
plt.savefig("node_evolution_all_layers_tsne.pdf", format="pdf", dpi=300)
plt.show()
