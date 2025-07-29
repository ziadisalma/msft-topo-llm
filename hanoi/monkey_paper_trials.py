from monkey_paper import make_fewshot_prompt

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

models = ["gpt2", "meta-llama/Meta-Llama-3-8B-Instruct"]
model = models[1]

tok = AutoTokenizer.from_pretrained(model)
mdl = AutoModelForCausalLM.from_pretrained(model).eval()

prompt, truth, cond = make_fewshot_prompt()
inputs = tok(prompt, return_tensors="pt")
with torch.no_grad():
    out = mdl.generate(**inputs, max_new_tokens=1)
pred = tok.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

print(prompt)
print("Model said:", pred, "| truth =", truth)
