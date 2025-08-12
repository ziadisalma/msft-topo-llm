import os
import gc
import json
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("HF_HOME", "/lambda/nfs/microsoft-rips2025/pzhang/.cache/")
os.environ.setdefault("TRANSFORMERS_CACHE", "/lambda/nfs/microsoft-rips2025/pzhang/.cache/")

MODEL_NAME = "microsoft/phi-4-reasoning"

def make_save_root(model_name: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    root = Path("/lambda/nfs/microsoft-rips2025/pzhang") / "phi4_r_attn_by_layer_head" / f"{model_name.replace('/', '_')}_{ts}"
    root.mkdir(parents=True, exist_ok=True)
    return root

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="eager",
)

# Capture attentions via hooks, not via aggregated outputs
model.config.output_attentions = False
model.config.output_hidden_states = True
model.config.use_cache = True

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

def denormalize_token_list(tokenizer, words):
    denormalized_tokens = []
    for word in words:
        tokens = tokenizer.tokenize(f" {word}")
        if tokens:
            denormalized_tokens.append(tokens[0])
    return denormalized_tokens

def generate_with_outputs(system_prompt, user_prompt, max_new_tokens=1024, temperature=0.6, top_p=0.9):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt_with_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_with_template, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )[0]

    with torch.no_grad():
        full_outputs = model(
            output_ids.unsqueeze(0),
            output_attentions=False, # attentions will be captured separately via hooks
            output_hidden_states=True,
            use_cache=False,
        )

    embeddings = full_outputs.hidden_states
    response_ids = output_ids[prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    del response_ids, inputs, full_outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return response_text, embeddings, output_ids

def _find_attention_modules(model):
    """
    Broad, order-preserving search for attention-like submodules.
    We do NOT require num_heads attributes (Phi-4 may not expose them).
    """
    candidates = []
    for name, mod in model.named_modules():
        lname = name.lower()
        cls = mod.__class__.__name__.lower()
        looks_like_attn = ("attn" in lname) or ("attention" in lname) or ("attn" in cls) or ("attention" in cls)
        if looks_like_attn:
            candidates.append((name, mod))
    # Convert to (1-based index, module, name)
    indexed = [(i + 1, m, n) for i, (n, m) in enumerate(candidates)]
    return indexed

def dump_attentions_by_layer_head(
    model,
    input_ids: torch.Tensor,
    save_root,
    save_dtype: torch.dtype = torch.float16,  # change to float32 if you prefer exact values
):
    """
    Runs one forward pass with output_attentions=True.
    Hooks each attention-like module. For each layer, we find the 4D attention tensor,
    save each head as a full (Q, K) matrix, and replace that element in the output
    tuple with None to avoid upstream aggregation.
    Prints a line after every saved head file and a short per-layer summary.
    """
    import json, gc, torch
    from pathlib import Path

    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    index_path = save_root / "index.json"

    # Minimal index written up-front so downstream readers don't crash
    seq_len = int(input_ids.shape[-1])
    meta = {
        "model_name": MODEL_NAME,
        "seq_len": seq_len,
        "layers": [],
        "dtype": str(save_dtype),
    }
    with open(index_path, "w") as f:
        json.dump(meta, f, indent=2)

    layer_specs = _find_attention_modules(model)
    hooks = []

    def _locate_attn_tensor_from_output(out):
        """
        Return (idx, tensor) where idx is the position inside a tuple/list 'out'
        that holds a 4D (B, H, Q, K) attention tensor. If not found, return (None, None).
        Preference: the largest 4D tensor by numel.
        """
        best = (None, None, -1)  # (idx, tensor, numel)
        if isinstance(out, (tuple, list)):
            for i, elem in enumerate(out):
                if torch.is_tensor(elem) and elem.dim() == 4:
                    b, h, q, k = elem.shape
                    if b >= 1 and h >= 1 and q >= 1 and k >= 1:
                        numel = elem.numel()
                        if numel > best[2]:
                            best = (i, elem, numel)
        return best[0], best[1]

    def make_hook(layer_idx: int, layer_name: str):
        def hook(_module, _inp, out):
            attn_idx, attn_probs = _locate_attn_tensor_from_output(out)
            if attn_idx is None or attn_probs is None:
                return out  # Not this module

            if attn_probs.dim() != 4:
                return out

            bsz, heads, q_len, k_len = attn_probs.shape
            if bsz != 1:
                return out  # this pipeline assumes batch size 1

            layer_dir = save_root / f"layer_{layer_idx:02d}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            # Record metadata for this layer once
            meta["layers"].append({
                "layer_index": layer_idx,
                "layer_name": layer_name,
                "num_heads": int(heads),
                "q_len": int(q_len),
                "k_len": int(k_len),
                "dir": str(layer_dir.relative_to(save_root)),
            })

            print(f"[attn-dump] Layer {layer_idx:02d} '{layer_name}': heads={heads}, QxK={q_len}x{k_len}", flush=True)

            saved_files = 0
            with torch.no_grad():
                for h in range(heads):
                    head_mat = attn_probs[0, h]   # (Q, K) on device
                    head_cpu = head_mat.to(dtype=save_dtype, device="cpu").contiguous()
                    shard_path = layer_dir / f"L{layer_idx:02d}_H{h:02d}.pt"
                    torch.save(
                        {
                            "layer": layer_idx,
                            "layer_name": layer_name,
                            "head": h,
                            "shape": (int(q_len), int(k_len)),
                            "dtype": str(save_dtype),
                            "attn": head_cpu,     # (Q, K)
                        },
                        shard_path
                    )
                    print(f"  [attn-dump] saved -> {shard_path}  (QxK={q_len}x{k_len}, dtype={save_dtype})", flush=True)
                    saved_files += 1
                    del head_cpu

            # Persist/refresh index after this layer
            with open(index_path, "w") as f:
                json.dump(meta, f, indent=2)

            print(f"[attn-dump] Finished layer {layer_idx:02d}: wrote {saved_files} head files to {layer_dir}", flush=True)

            # Prevent upstream accumulation: replace the attention tensor with None
            if isinstance(out, (tuple, list)):
                out_list = list(out)
                out_list[attn_idx] = None
                return tuple(out_list)
            return out
        return hook

    # Register hooks (fallback to all modules if none matched explicitly)
    if len(layer_specs) == 0:
        for i, (name, module) in enumerate(model.named_modules(), start=1):
            hooks.append(module.register_forward_hook(make_hook(i, name)))
    else:
        for (layer_idx, module, layer_name) in layer_specs:
            hooks.append(module.register_forward_hook(make_hook(layer_idx, layer_name)))

    try:
        with torch.no_grad():
            _ = model(
                input_ids.unsqueeze(0).to(model.device),
                output_attentions=True,   # triggers attention computation; hooks do the saving
                output_hidden_states=False,
                use_cache=False,
            )
    finally:
        for h in hooks:
            h.remove()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "save_root": str(save_root),
        "index_path": str(index_path),
        "num_layers_found": len(layer_specs),
        "num_layers_captured": len(meta["layers"]),
        "seq_len": seq_len,
    }

def process_embeddings(embeddings, output_ids, tokenizer, target_tokens, layers=None, as_numpy=True):
    seq_tokens = tokenizer.convert_ids_to_tokens(output_ids)
    positions = [i for i, t in enumerate(seq_tokens) if t in target_tokens]
    if not positions: return {}

    total_layers = len(embeddings) - 1
    layer_indices = layers if layers is not None else list(range(1, total_layers + 1))

    processed_embs = {}
    for l in layer_indices:
        if l < 1 or l >= len(embeddings): continue

        layer_hs = embeddings[l][0]  # (seq_len, hidden_size)
        selected = layer_hs[positions]
        if as_numpy:
            selected = selected.to(torch.float32).cpu().numpy()

        processed_embs[l] = {
            'tokens': [seq_tokens[i] for i in positions],
            'positions': positions,
            'embeddings': selected
        }
    return processed_embs

def process_attentions_from_files(attn_root_dir, output_ids, tokenizer, target_tokens, query_position=-1, layers=None, head_average=True, as_numpy=True):
    """
    Reads per-head attention files (one (Q, K) matrix per head) and returns
    scores for a single query position to selected key positions.
    """
    import json, torch
    from pathlib import Path

    attn_root_dir = Path(attn_root_dir)
    index_path = attn_root_dir / "index.json"
    if not index_path.exists():
        print(f"[process_attentions_from_files] No index.json at {index_path}. Did the dump step capture any layers?")
        return {}

    with open(index_path, "r") as f:
        meta = json.load(f)

    seq_tokens = tokenizer.convert_ids_to_tokens(output_ids)
    key_positions = [i for i, t in enumerate(seq_tokens) if t in target_tokens]
    if not key_positions:
        return {}

    layer_map = {layer["layer_index"]: layer for layer in meta.get("layers", [])}
    available_layers = sorted(layer_map.keys())
    layer_indices = layers if layers is not None else available_layers

    if not available_layers:
        print(f"[process_attentions_from_files] Index has no layers. Root={attn_root_dir}")
        return {}

    # Normalize query position
    q_len_global = meta.get("seq_len", len(seq_tokens))
    q_pos = query_position if query_position >= 0 else (len(seq_tokens) + query_position)
    if q_pos < 0 or q_pos >= q_len_global:
        return {}

    processed_attns = {}
    for l in layer_indices:
        if l not in layer_map:
            continue
        info = layer_map[l]
        layer_dir = attn_root_dir / info["dir"]
        heads = info["num_heads"]

        accum = None
        for h in range(heads):
            path = layer_dir / f"L{l:02d}_H{h:02d}.pt"
            if not path.exists():
                # Be permissive if some heads weren't written
                continue
            shard = torch.load(path, map_location="cpu")
            attn = shard["attn"]  # (Q, K)
            row = attn[q_pos, key_positions].to(torch.float32)  # (len(keys),)
            if accum is None:
                accum = row
            else:
                accum += row

        if accum is None:
            continue
        if head_average and heads > 0:
            accum = accum / float(heads)
        if as_numpy:
            accum = accum.numpy()

        processed_attns[l] = {
            'query_token': seq_tokens[q_pos],
            'key_tokens': [seq_tokens[i] for i in key_positions],
            'scores': accum
        }

    return processed_attns

if __name__ == "__main__":
    if not torch.cuda.is_available():
        model.to(torch.float32)

    sys_prompt  = "You are a concise, knowledgeable assistant."
    user_prompt = "Explain the difference between a list and a tuple in Python in two sentences."

    reply, embeddings, output_ids = generate_with_outputs(
        sys_prompt,
        user_prompt,
        max_new_tokens=512,
    )
    print(f"System prompt: {sys_prompt}\nUser prompt: {user_prompt}\nAssistant reply: {reply}")

    # Save attentions (layer/head granularity)
    save_root = make_save_root(MODEL_NAME)
    print(f"Saving attention (layer/head) to: {save_root}")

    dump_info = dump_attentions_by_layer_head(
        model=model,
        input_ids=output_ids,
        save_root=save_root,
        save_dtype=torch.float16,
    )
    print(f"Attention dump complete. Index: {dump_info['index_path']}")
    print(f"Layers matched by finder: {dump_info['num_layers_found']}  "
          f"Layers actually captured: {dump_info['num_layers_captured']}  "
          f"Seq len: {dump_info['seq_len']}")

    # Token targets and embedding extraction (unchanged)
    TARGETS = denormalize_token_list(tokenizer, ['list', 'lists', 'tuple', 'tuples'])

    specific_embeddings = process_embeddings(
        embeddings, output_ids, tokenizer,
        target_tokens=TARGETS,
        layers=[1, 16, 32],
    )
    for layer, info in specific_embeddings.items():
        print(f"  Layer {layer}: Found tokens {info['tokens']} at positions {info['positions']}. "
              f"Embedding shape: {info['embeddings'].shape}")

    # Read attention back from disk for selected query/key tokens
    specific_attentions = process_attentions_from_files(
        attn_root_dir=save_root,
        output_ids=output_ids,
        tokenizer=tokenizer,
        target_tokens=TARGETS,
        query_position=-1,  # last token
        layers=[1, 16, 32],
    )
    for layer, info in specific_attentions.items():
        print(f" Layer {layer}: Attention from final token ('{info['query_token']}') to {info['key_tokens']}:")
        print(f" Scores: {info['scores']}")

