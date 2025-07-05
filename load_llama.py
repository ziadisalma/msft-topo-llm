import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import utilities.authentication

hf_token = utilities.authentication.get_hf_token()
login(token=hf_token)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
torch_dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    device_map="auto"
)

def generate_response(system_prompt, user_prompt, max_new_tokens = 512) -> str:
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
            temperature=0.6,
            top_p=0.9,
        )

    response_ids = outputs[0][inputs['input_ids'].shape[-1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    del outputs
    del response_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return response

if __name__ == "__main__":
    system_prompt = "You are a helpful AI assistant."
    user_prompt = "What is the capital of France and why is it famous?"

    print("Generating response...")
    model_response = generate_response(system_prompt, user_prompt)
    print("\n--- Model Response ---")
    print(model_response)
    print("----------------------\n")

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
