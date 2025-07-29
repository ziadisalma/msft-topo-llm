import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
torch_dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    device_map="auto",
    attn_implementation="eager"
)

model.config.output_attentions = True
model.config.use_cache = True
tokenizer.pad_token_id = tokenizer.eos_token_id

def generate_response(system_prompt, user_prompt, max_new_tokens=512, temperature=0.6, top_p=0.9):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt_with_template = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt_with_template, return_tensors="pt").to(model.device)
    eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    response_ids = outputs[0][inputs['input_ids'].shape[-1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    # cleanup
    del outputs
    del response_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return response

def extract_token_embeddings(text, tokens=None, layers=None, as_numpy=True):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Forward pass to get all hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states

    # Determine layers: skip embedding layer at index 0
    total_layers = len(hidden_states) - 1
    layer_indices = layers if layers is not None else list(range(1, total_layers + 1))

    # Convert input IDs to tokens
    seq_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Determine token positions to extract
    if tokens is not None:
        positions = [i for i, t in enumerate(seq_tokens) if t in tokens]
    else:
        positions = list(range(len(seq_tokens)))

    embeddings = {}
    for l in layer_indices:
        layer_hs = hidden_states[l][0]  # shape: (seq_len, hidden_size)
        selected = layer_hs[positions]  # shape: (len(positions), hidden_size)
        if as_numpy:
            # cast bfloat16 to float32 for numpy conversion
            selected = selected.to(torch.float32).cpu().numpy()
        embeddings[l] = {
            'tokens': [seq_tokens[i] for i in positions],
            'positions': positions,
            'embeddings': selected
        }

    # cleanup
    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return embeddings

def extract_attention_to_tokens(system_prompt, user_prompt, tokens=None, layers=None, max_new_tokens=1, head_average=True, as_numpy=True):
    # Build and tokenize chat prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate answer-token(s)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )[0]
    prompt_len = inputs["input_ids"].shape[-1]
    prompt_ids = gen_ids[:prompt_len]
    answer_ids = gen_ids[prompt_len:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    # Forward pass over full sequence to capture attentions
    prev_cache = model.config.use_cache
    model.config.use_cache = False
    model.config.output_attentions = True
    with torch.no_grad():
        out = model(gen_ids.unsqueeze(0), output_attentions=True, return_dict=True)
    attn = out.attentions
    model.config.use_cache = prev_cache

    # Tokens and positions
    seq_tokens = tokenizer.convert_ids_to_tokens(prompt_ids)
    if tokens is not None:
        positions = [i for i, t in enumerate(seq_tokens) if t in tokens]
    else:
        positions = list(range(len(seq_tokens)))
    labels = [seq_tokens[i] for i in positions]

    # Layers
    total_layers = len(attn)
    layer_indices = layers if layers is not None else list(range(1, total_layers+1))

    # Build attention map
    attentions: dict[int, torch.Tensor|np.ndarray] = {}
    for L in layer_indices:
        # attn[L-1]: (batch=1, heads, seq, seq)
        layer_attn = attn[L-1][0]  # heads, seq, seq
        if head_average:
            mean_attn = layer_attn.mean(dim=0)  # seq, seq
            vec = mean_attn[-1, positions]
        else:
            vec = layer_attn[:, -1, positions]  # heads, positions
        if as_numpy:
            vec = vec.to(torch.float32).cpu().numpy()
        attentions[L] = vec

    # cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        'answer':     answer,
        'tokens':     labels,
        'positions':  positions,
        'layers':     layer_indices,
        'attentions': attentions,
    }

