import random
import string
import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



# ─ PARAMETERS ────────────────────────────────────────────────────────────────
NUM_GRAPHS   = 20    # total number of graphs to generate
DIAMETER     = 4        # target diameter for all graphs
TRANSPARENCY = 95       # as in your template; will map to temperature = 0.05
#SEED         = 1
START_ROOM   = 'lobby'
MODEL_NAME   = "meta-llama/Meta-Llama-3.1-8B-Instruct"

#random.seed(SEED)
#torch.manual_seed(SEED)

# ─ GRAPH GENERATION ──────────────────────────────────────────────────────────
def generate_tree_with_diameter(n_nodes, diameter):
    """
    Build a tree on `n_nodes` nodes whose diameter is exactly `diameter`.
    We start with a path of length diameter, then attach extra nodes randomly.
    """
    G = nx.path_graph(diameter + 1)
    next_node = diameter + 1
    while next_node < n_nodes:
        attach_to = random.choice(list(G.nodes()))
        G.add_node(next_node)
        G.add_edge(attach_to, next_node)
        next_node += 1
    return G

def generate_graph(diameter, is_cyclic):
    """
    Generate either a tree (is_cyclic=False) or
    a cyclic graph (is_cyclic=True) of the given diameter.
    """
    # ensure at least diameter+1 nodes; add a couple extra for variety
    n_nodes = diameter + 3
    G = generate_tree_with_diameter(n_nodes, diameter)
    if is_cyclic:
        # add one random extra edge to introduce exactly one cycle
        while True:
            u, v = random.sample(G.nodes(), 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                break
    return G

def graph_to_description(G, start_label):
    """
    Convert graph G into a simple NL description,
    where node 0 is 'start_label' and others get assigned letters.
    We do a BFS from node 0 and describe each forward adjacency.
    """
    # assign labels
        # 1) Build a shuffled list of labels
    all_letters = list(string.ascii_uppercase)
    all_letters.remove(start_label[0].upper())   # reserve the lobby letter
    random.shuffle(all_letters)

    # 2) Assign labels by zipping nodes → letters
    mapping = {node: label for node, label in zip(G.nodes(), [start_label] + all_letters)}


    # BFS description
    lines = []
    for u, v in G.edges():
        lines.append(f"{mapping[u]} connects to {mapping[v]}")
    description = "From the lobby, " + ". ".join(lines) + "."
    return description

# ─ BUILD EXAMPLES ────────────────────────────────────────────────────────────
examples = []
for i in range(NUM_GRAPHS):
    is_cycle = (i % 2 == 0)  # alternate tree / cyclic
    G = generate_graph(DIAMETER, is_cycle)
    sentence = graph_to_description(G, START_ROOM)
    # record (description, correct answer)
    examples.append((sentence, "Yes" if is_cycle else "No"))

# ─ PREPARE PROMPTS ───────────────────────────────────────────────────────────
request_template = (
    "Given the description of rooms below, determine if there is a cycle in the undirected map. "
     "Show your steps then add a final line 'Answer: Yes' or 'Answer: No'. Do not repeat the prompt. Keep your answer under 128 tokens.\n\n"
    "Description:\n{}\n"
    "Answer: "
)
prompts = [request_template.format(desc) for desc, _ in examples]

# ─ LOAD LLAMA MODEL ──────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

# ─ INFERENCE ─────────────────────────────────────────────────────────────────
temperature = (100 - TRANSPARENCY) / 100  # e.g. TRANSPARENCY=95 → temp=0.05
results = []

for prompt, (_, correct) in zip(prompts, examples):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=temperature,
    )
    reply = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    print(f"\n\nPrompt:\n{prompt}\nModel → {reply} (expected: {correct})\n")
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0, prompt_len:]     
    nntokens = gen_ids.size(0)
    print("num tokens:", nntokens)

    results.append((prompt, reply, correct))
