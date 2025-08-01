import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "microsoft/Phi-4-reasoning"

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
        tokens = tokenizer.tokenize(f" {word}")
        if tokens:
            denormalized_tokens.append(tokens[0])
    return denormalized_tokens

def generate_with_outputs(system_prompt, user_prompt, max_new_tokens=1024, temperature=0.8, top_p=0.95, top_k=50):
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
        top_k=top_k,
    )[0]

    with torch.no_grad():
        full_outputs = model(
            output_ids.unsqueeze(0),
            output_attentions=True,
            output_hidden_states=True
        )

    attentions = full_outputs.attentions
    embeddings = full_outputs.hidden_states
    
    response_ids = output_ids[prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    del response_ids, inputs, full_outputs
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
        
        layer_hs = embeddings[l][0]
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

        layer_attn = attentions[l-1][0]
        vec = layer_attn[:, query_position, key_positions]
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

    sys_prompt = "You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:"
    user_prompt = "Explain the difference between a list and a tuple in Python in two sentences."

    reply, attentions, embeddings, output_ids = generate_with_outputs(
        sys_prompt,
        user_prompt,
        max_new_tokens=4096,
        temperature=0.8,
        top_p=0.95,
        top_k=50
    )

    print(f"System prompt: {sys_prompt}\n\nUser prompt: {user_prompt}\n\nAssistant reply:\n{reply}")
    
    TARGETS = denormalize_token_list(tokenizer, ['list', 'lists', 'tuple', 'tuples', 'mutable', 'immutable'])

    specific_embeddings = process_embeddings(
        embeddings, output_ids, tokenizer,
        target_tokens=TARGETS,
        layers=[1, 16, 32],
    )
    print("\n--- Processed Embeddings ---")
    for layer, info in specific_embeddings.items():
        print(f"  Layer {layer}: Found tokens {info['tokens']} at positions {info['positions']}. Embedding shape: {info['embeddings'].shape}")

    specific_attentions = process_attentions(
        attentions, output_ids, tokenizer,
        target_tokens=TARGETS,
        query_position=-1,
        layers=[1, 16, 32],
    )
    print("\n--- Processed Attentions ---")
    for layer, info in specific_attentions.items():
        print(f"  Layer {layer}: Attention from final token ('{info['query_token']}') to {info['key_tokens']}:")
        print(f"    Scores: {info['scores']}")
