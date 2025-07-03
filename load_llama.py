import os
import gc
import torch
import utilities.authentication
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

hf_token = utilities.authentication.get_hf_token()
MODEL_NAME = "meta-llama/Llama-3.2-3B"

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
    print(f"Input >>>: {prompt}")
    inputs = encode_prompt(prompt)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    input_len = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.10,
            temperature=1.0,
            top_p=1.0
        )

    gen_ids = output_ids[0][input_len:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    print(f"Output <<<: {response}")
    return response

