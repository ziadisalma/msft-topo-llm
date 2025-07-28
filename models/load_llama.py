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
    device_map="auto"
)

def generate_response(system_prompt, user_prompt, max_new_tokens=512, temperature=0.6, top_p=0.9) -> str:
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
