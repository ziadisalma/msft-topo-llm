import os
import gc
import torch
import utilities.authentication
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

hf_token = utilities.authentication.get_hf_token()
MODEL_NAME = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    token=hf_token
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
model.eval()

def encode_prompt(prompt: str):
    return tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

def generate_response(prompt: str, max_new_tokens: int = 64):
    inputs = encode_prompt(prompt)
    device = next(model.parameters()).device
    batch = inputs["input_ids"].shape
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # compute absolute max_length = input_length + new tokens
    input_len = inputs["input_ids"].shape[-1]
    max_length = input_len + max_new_tokens

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

def main():
    prompt = "Hello, Llama! How are you today?"
    print(f">>> Prompt: {prompt}\n")

    response = generate_response(prompt)
    print(f"<<< Response: {response}\n")

if __name__ == "__main__":
    main()
