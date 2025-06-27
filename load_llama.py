import gc
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

MODEL_NAME = "meta-llama/Llama-3-7b-chat-hf"

tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
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
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    # decode into text
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # GPU & Python GC cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return response
