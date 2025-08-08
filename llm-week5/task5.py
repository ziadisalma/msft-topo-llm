
# -----------------------------------
import random,textwrap
from pathlib import Path
import networkx as nx                
from transformers import AutoTokenizer, AutoModelForCausalLM  

# ---------- Graph‑generation utilities ----------
def generate_graph(n_range=(6, 10), edge_factor=1.5, w_range=(1, 20), seed=None):
    """Return (edges, start, goal).  Ensures at least one start→goal path."""
    rng = random.Random(seed)
    n = rng.randint(*n_range)
    nodes = [chr(65+i) for i in range(n)]
    start, goal = rng.sample(nodes, 2)

    # guaranteed backbone path start→…→goal
    k = rng.randint(2, max(2, n // 2))
    middle = rng.sample([x for x in nodes if x not in (start, goal)], k-1)
    backbone = [start] + middle + [goal]

    edges = [(u, v, rng.randint(*w_range)) for u, v in zip(backbone, backbone[1:])]

    # extra edges
    want_m = int(edge_factor * n)
    while len(edges) < want_m:
        u, v = rng.sample(nodes, 2)
        if (u, v) not in {(a, b) for a, b, _ in edges}:
            edges.append((u, v, rng.randint(*w_range)))
    return edges, start, goal

def edge_list_to_text(edges):
    return "\n".join(f"{u} -> {v} ({w})" for u, v, w in edges)

message = """Your task is to compute the TOTAL WEIGHT of the MINIMUM-COST path between a start node and a goal node.
   The minimum cost path involves finding the path from the source node to the destination node that minimizes the total cost.


   ### Graph (directed, weighted)
   Each line: FROM -> TO (weight)
   {edge_list}


   ### Task (Total‑cost variant)
   • Start node: {s}
   • Goal node : {g}


   ➜ Output the total weight in this format: The total weight is 'INT_TOTAL_WEIGHT'. """



# ---------- Main batch runner ----------
def main():
    NUM_GRAPHS = 1
    MODEL_NAME = "microsoft/Phi-4-reasoning"  
    MAX_NEW = 4096

    # load model/tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                 torch_dtype="auto",
                                                 device_map="auto")


    for idx in range(NUM_GRAPHS):
        edges, s, g = generate_graph(seed=None)  
        fmessage = message.format(edge_list=edge_list_to_text(edges), s=s, g=g)
        messages = [
        {"role": "system", "content": "You are an expert graph‑algorithm assistant."},
        {"role": "user", "content": fmessage},
]
        # LLM inference
        inp = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors="pt")
        out_ids = model.generate(inp.to(model.device), max_new_tokens=MAX_NEW, temperature = 0.8, top_k=50, top_p=0.95, do_sample=True)
        reply = tokenizer.decode(out_ids[0])

        # ground truth via NetworkX
        G = nx.DiGraph(); G.add_weighted_edges_from(edges)
        true_cost = nx.shortest_path_length(G, s, g, weight="weight")

        #print model output and ground_truth
        print("\n\n", fmessage, "\n\n")
        print("***Model's output*** \n", reply, "\n\n")
        print("***exptected resukt***: ", true_cost)

if __name__ == "__main__":
    main()

