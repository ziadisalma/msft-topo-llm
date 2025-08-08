#!/usr/bin/env python3
"""
batch_shortest_path_swap.py  –  100 random graphs with a weight-swap twist
"""

import random, heapq
import re
import torch
from itertools import combinations

# -------- config --------
NUM_TRIALS, MIN_NODES, MAX_NODES, EXTRA_EDGES = 100, 6, 10, 3
WEIGHT_LO, WEIGHT_HI = 1, 25
RUN_MODEL = True            # flip True if you want to query a model
MODEL_NAME, TEMPERATURE = "microsoft/Phi-4", 0.8
# ------------------------

def canon(u, v):
    return (u, v) if u < v else (v, u)

def make_connected_graph():
    n = random.randint(MIN_NODES, MAX_NODES)
    nodes = [chr(ord("A") + i) for i in range(n)]
    edges = set()

    # (1) spanning tree
    for i in range(1, n):
        u, v = nodes[i], random.choice(nodes[:i])
        w = random.randint(WEIGHT_LO, WEIGHT_HI)
        edges.add((*canon(u, v), w))                       # FIX ⬅

    # (2) extra edges
    possible = list(combinations(nodes, 2))
    random.shuffle(possible)
    added = 0
    for u, v in possible:
        if added >= EXTRA_EDGES:
            break
        if not any({u, v}.issubset({e[0], e[1]}) for e in edges):
            w = random.randint(WEIGHT_LO, WEIGHT_HI)
            edges.add((*canon(u, v), w))                   # FIX ⬅
            added += 1
    return nodes, list(edges)

def dijkstra(nodes, edges, src, dst):
    adj = {v: [] for v in nodes}
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    dist, prev = {v: float('inf') for v in nodes}, {v: None for v in nodes}
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        if u == dst: break
        for v, w in adj[u]:
            alt = d + w
            if alt < dist[v]:
                dist[v], prev[v] = alt, u
                heapq.heappush(pq, (alt, v))
    path, cur = [], dst
    while cur is not None:
        path.append(cur); cur = prev[cur]
    return dist[dst], path[::-1]          # cost, node-list

def swap_weights(edges, e1, e2):
    (u1, v1, w1), (u2, v2, w2) = e1, e2
    out = []
    for u, v, w in edges:
        if {u, v} == {u1, v1}: out.append((u, v, w2))
        elif {u, v} == {u2, v2}: out.append((u, v, w1))
        else: out.append((u, v, w))
    return out

# optional model loader
if RUN_MODEL:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
    def ask_model(p):        
        prompt_with_template = tokenizer.apply_chat_template(p, tokenize = False, add_generation_prompt = True)
        input = tokenizer(prompt_with_template, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**input, max_new_tokens=2048, eos_token_id=tokenizer.eos_token_id, temperature=TEMPERATURE, top_k=50, top_p=0.95, do_sample=True)
        response_ids = out_ids[0][input["input_ids"].shape[-1]:]
        return tokenizer.decode(response_ids, skip_special_tokens=True)
    

# -------- main loop --------
for t in range(1, NUM_TRIALS + 1):
    nodes, edges = make_connected_graph()
    start, goal  = random.sample(nodes, 2)
    cost0, path  = dijkstra(nodes, edges, start, goal)

    # edge on the path / edge off the path
    path_edges = {canon(path[i], path[i+1]) for i in range(len(path)-1)}
    edge1 = random.choice([e for e in edges if canon(e[0], e[1]) in path_edges])
    edge2 = random.choice([e for e in edges if canon(e[0], e[1]) not in path_edges])

    edges_swapped = swap_weights(edges, edge1, edge2)
    cost_swapped, _ = dijkstra(nodes, edges_swapped, start, goal)

    # prompt
    e1, e2 = edge1, edge2
    messages = [
    {"role": "system", "content":"You are an expert at solving graph theory problems, specifically finding shortest paths in weighted undirected graphs."},
    {"role": "user", "content": f"""Your task is to compute the **total weight of the shortest path** between Start and Goal in the **UNDIRECTED** weighted graph below *after swapping the two highlighted edges*.

Edge 1 : {e1[0]} -- {e1[1]} (weight {e1[2]})  
Edge 2 :          {e2[0]} -- {e2[1]} (weight {e2[2]})  
After the swap, their weights are exchanged.

### Graph
Each line: FROM -- TO (weight)
{chr(10).join(f"{u} -- {v} ({w})" for u, v, w in edges)}

Start node: {start}  
Goal node : {goal}

➜ Give your answer like this: The total weight is INT_TOTAL_WEIGHT
"""}]

    print(f"\n========== TRIAL {t:03d} PROMPT ==========")
    print(messages[1])
    print(f"Ground truth: {cost_swapped}")

    if RUN_MODEL:
        ans = ask_model(messages)
        print("\n-- Model response --")
        print(ans)
        m = re.findall(r"\d+", ans)          # all digit groups
        pred = int(m[-1]) if m else None      # take the last one
        ok   = (pred == cost_swapped)
        print(f"[Model {'✔' if ok else '✘'}]  (predicted {pred}, truth {cost_swapped})")
