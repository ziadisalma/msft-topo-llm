from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re, statistics
import json
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
)


def chat(messages, temperature=0.5):
        input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
        ).to(model.device)

        output = model.generate(
                input_ids,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ],
         )[0]

        gen_ids = output[input_ids.shape[-1] :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)      
        num_new_tokens = len(gen_ids)
        return text, num_new_tokens



prompt = """Imagine you are in a hotel with rooms denotated by letters and connected by two way doors. Room G is connected to room R, room D, and room W. Room W is also connected to room D and room J. Room R is also connected to room L.
"""

with open("definitions.json", "r") as f:
    definitions = json.load(f)

needed_metric = "Average shortest path"
definition = definitions[needed_metric]

convo = [
    {"role": "system", "content": "You are a graph analyst."},
    {"role": "user", "content": prompt},
    {"role": "user", "content": definition},
    {"role": "user", "content": "Let's think step by step."}
]


K = 10  
new_token_counts = []

for _ in range(K):
        answer, n_tokens = chat(convo, temperature=0.5)
        print("\n", answer, "\n")
        new_token_counts.append(n_tokens)

print(f"Average generated tokens on reval: {statistics.mean(new_token_counts):.1f}")
