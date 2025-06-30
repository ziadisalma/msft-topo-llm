import torch

def generate_one_token(prompt: str):
    inputs = encode_prompt(prompt)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
        )
    new_tokens = output_ids[0, input_ids.shape[1]:]
    token_str = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return token_str
