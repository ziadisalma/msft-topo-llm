import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "microsoft/phi-4"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="eager",
)

model.config.output_attentions = True
model.config.output_hidden_states = True
model.config.use_cache = True
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

def denormalize_token_list(tokenizer, words):
    denormalized_tokens = []
    for word in words:
        # Tokenize the word with a preceding space to get the internal representation.
        # We take the first result as words can sometimes be split into multiple tokens.
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

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )[0]

    prev_cache_setting = model.config.use_cache
    model.config.use_cache = False
    with torch.no_grad():
        outputs = model(output_ids.unsqueeze(0), return_dict=True)
    model.config.use_cache = prev_cache_setting

    response_ids = output_ids[prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    attentions = outputs.attentions
    embeddings = outputs.hidden_states

    # cleanup
    del outputs, response_ids, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return response_text, attentions, embeddings, output_ids

def process_embeddings(embeddings, output_ids, tokenizer, target_tokens, layers=None, as_numpy=True):
    seq_tokens = tokenizer.convert_ids_to_tokens(output_ids)
    positions = [i for i, t in enumerate(seq_tokens) if t in target_tokens]
    if not positions: return {}

    total_layers = len(embeddings) - 1
    layer_indices = layers if layers is not None else list(range(1, total_layers + 1))

    processed_embs = {}
    for l in layer_indices:
        if l < 1 or l >= len(embeddings): continue
        
        layer_hs = embeddings[l][0] # (seq_len, hidden_size)
        selected = layer_hs[positions]
        if as_numpy:
            selected = selected.to(torch.float32).cpu().numpy()
            
        processed_embs[l] = {
            'tokens': [seq_tokens[i] for i in positions],
            'positions': positions,
            'embeddings': selected
        }
    return processed_embs

def process_attentions(attentions, output_ids, tokenizer, target_tokens, query_position=-1, layers=None, head_average=True, as_numpy=True):
    seq_tokens = tokenizer.convert_ids_to_tokens(output_ids)
    key_positions = [i for i, t in enumerate(seq_tokens) if t in target_tokens]
    if not key_positions: return {}

    total_layers = len(attentions)
    layer_indices = layers if layers is not None else list(range(1, total_layers + 1))
    
    processed_attns = {}
    for l in layer_indices:
        if l < 1 or l > total_layers: continue

        layer_attn = attentions[l-1][0] # Use l-1 for 0-based tensor index
        vec = layer_attn[:, query_position, key_positions] # (heads, key_positions)
        if head_average:
            vec = vec.mean(dim=0)
        if as_numpy:
            vec = vec.to(torch.float32).cpu().numpy()

        processed_attns[l] = {
            'query_token': seq_tokens[query_position],
            'key_tokens': [seq_tokens[i] for i in key_positions],
            'scores': vec
        }
    return processed_attns

if __name__ == "__main__":
    if not torch.cuda.is_available():
        model.to(torch.float32)

    sys_prompt  = "You are a concise, knowledgeable assistant."
    user_prompt = "Explain the difference between a list and a tuple in Python in two sentences."

    reply, attentions, embeddings, output_ids = generate_with_outputs(
        sys_prompt,
        user_prompt,
        max_new_tokens=128,
    )

    print(f"System prompt: {sys_prompt}\nUser prompt: {user_prompt}\nAssistant reply: {reply}")
    
    TARGETS = denormalize_token_list(tokenizer, ['list', 'lists', 'tuple', 'tuples'])

    specific_embeddings = process_embeddings(
        embeddings, output_ids, tokenizer,
        target_tokens=TARGETS,
        layers=[1, 16, 32], # Example layers
    )
    for layer, info in specific_embeddings.items():
        print(f"  Layer {layer}: Found tokens {info['tokens']} at positions {info['positions']}. Embedding shape: {info['embeddings'].shape}")

    specific_attentions = process_attentions(
        attentions, output_ids, tokenizer,
        target_tokens=TARGETS,
        query_position=-1, # -1 means the very last token
        layers=[1, 16, 32],
    )
    for layer, info in specific_attentions.items():
        print(f"  Layer {layer}: Attention from final token ('{info['query_token']}') to {info['key_tokens']}:")
        print(f"    Scores: {info['scores']}")
