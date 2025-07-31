import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "microsoft/phi-4"
torch_dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    device_map="auto",
    attn_implementation="eager",
)

model.config.output_attentions = True
model.config.use_cache = True
tokenizer.pad_token_id = tokenizer.eos_token_id

def generate_response(system_prompt, user_prompt, max_new_tokens=1024, temperature=0.6, top_p=0.9):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    prompt_with_template = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True, # appends "<|im_start|>assistant<|im_sep|>"
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

    response_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    # cleanup
    del outputs, response_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return response

def extract_token_embeddings(text, tokens=None, layers=None, as_numpy=True):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states

    total_layers = len(hidden_states) - 1 # skip embedding layer (0)
    layer_indices = layers if layers is not None else list(range(1, total_layers + 1))

    seq_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    positions = (
        [i for i, t in enumerate(seq_tokens) if t in tokens] if tokens is not None
        else list(range(len(seq_tokens)))
    )

    embeddings = {}
    for l in layer_indices:
        layer_hs = hidden_states[l][0] # (seq_len, hidden_size)
        selected = layer_hs[positions]
        if as_numpy:
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

    # generate answer token(s)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )[0]

    prompt_len = inputs["input_ids"].shape[-1]
    prompt_ids  = gen_ids[:prompt_len]
    answer_ids  = gen_ids[prompt_len:]
    answer      = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    # capture attentions
    prev_cache = model.config.use_cache
    model.config.use_cache = False
    model.config.output_attentions = True
    with torch.no_grad():
        out = model(gen_ids.unsqueeze(0), output_attentions=True, return_dict=True)
    attn = out.attentions
    model.config.use_cache = prev_cache

    seq_tokens = tokenizer.convert_ids_to_tokens(prompt_ids)
    positions  = (
        [i for i, t in enumerate(seq_tokens) if t in tokens] if tokens else
        list(range(len(seq_tokens)))
    )
    labels = [seq_tokens[i] for i in positions]

    total_layers = len(attn)
    layer_indices = layers if layers is not None else list(range(1, total_layers + 1))

    attentions = {}
    for L in layer_indices:
        # attn[L-1]: (batch=1, heads, seq, seq)
        layer_attn = attn[L-1][0]
        if head_average:
            vec = layer_attn.mean(dim=0)[-1, positions] # (positions,)
        else:
            vec = layer_attn[:, -1, positions] # (heads, positions)
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

if __name__ == "__main__":
    if not torch.cuda.is_available():
        model.to(torch.float32)

    sys_prompt  = "You are a concise, knowledgeable assistant."
    user_prompt = "Explain the difference between a list and a tuple in Python in two sentences."
    reply = generate_response(sys_prompt, user_prompt, max_new_tokens=128)

    print(f"System prompt: {sys_prompt}\nUser prompt: {user_prompt}\nAssistant reply: {reply}")

    emb = extract_token_embeddings(
        reply,
        tokens=["lists", "tuples"],
        layers=[1, 2],
        as_numpy=True,
    )
    for layer, info in emb.items():
        print(f"Layer {layer}: shape {info['embeddings'].shape} for tokens {info['tokens']}")
    print()

    attn_out = extract_attention_to_tokens(
        sys_prompt,
        user_prompt,
        tokens=["list", "tuple"],
        layers=[1],
        max_new_tokens=1,
        head_average=True,
        as_numpy=True,
    )
    print("Generated token:", attn_out["answer"])
    for L, vec in attn_out["attentions"].items():
        print(f"Layer {L} attention vector: {vec}")
