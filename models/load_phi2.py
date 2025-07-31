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
model.config.output_hidden_states = True
model.config.use_cache = True
tokenizer.pad_token_id = tokenizer.eos_token_id

def generate_with_outputs(system_prompt, user_prompt, max_new_tokens=1024, temperature=0.6, top_p=0.9):
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
    del outputs, response_ids, output_ids, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return response_text, attentions, embeddings

if __name__ == "__main__":
    if not torch.cuda.is_available():
        model.to(torch.float32)

    sys_prompt  = "You are a concise, knowledgeable assistant."
    user_prompt = "Explain the difference between a list and a tuple in Python in two sentences."

    # Generate the response and extract outputs in a single, efficient call
    reply, attentions, embeddings = generate_with_outputs(
        sys_prompt,
        user_prompt,
        max_new_tokens=128,
    )

    print(f"System prompt: {sys_prompt}\nUser prompt: {user_prompt}\nAssistant reply: {reply}\n")

    # The 'embeddings' tuple contains the hidden states from each layer, plus the initial input embeddings.
    # The length is num_layers + 1.
    print(f"Extracted {len(embeddings)} embedding tensors (1 for input + {len(embeddings)-1} for layers).")
    
    # Shape: (batch_size=1, sequence_length, hidden_size)
    final_layer_embeddings = embeddings[-1]
    print(f"Shape of final layer embeddings: {final_layer_embeddings.shape}")

    # The length is num_layers.
    print(f"Extracted {len(attentions)} attention tensors (1 for each layer).")
    
    # Shape: (batch_size=1, num_heads, sequence_length, sequence_length)
    first_layer_attentions = attentions[0]
    print(f"Shape of first layer attentions: {first_layer_attentions.shape}")
