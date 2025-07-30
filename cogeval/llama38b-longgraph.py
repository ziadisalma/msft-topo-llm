# pip install transformers accelerate torch --upgrade
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re, statistics

# ------------------------------ SETUP ------------------------------
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,   # saves VRAM; use float16 if needed
        device_map="auto"
)

model.config.pad_token_id = tokenizer.pad_token_id 

k_runs   = 10          # how many completions you want
temperature = 0.5      
truth = 1              # correct room number for this toy graph

prompt = (
    """Imagine a world with 6 rooms. From the lobby you have two choices, room 1 and room 2.
You enter room 1, at the end there’s a door that leads to room 3, and room 3 leads to room 5.
There’s a chest in room 5. You open it and there’s 10 dollars. Then you exit and start over.
This time in the lobby you choose room 2, then enter room 4, which leads to room 6. There’s
a chest with 50 dollars. You return to the lobby. Which room will you choose to make the
most money? Only return room number""" 
)

# ------------------------------ LOOP ------------------------------
gen_tok_list = []

for _ in range(k_runs):
        messages = [
                {"role": "system", "content": "You are a helpful reasoning assistant."},
                {"role": "user",   "content": prompt},
        ]

        
        input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        
        attention_mask = torch.ones_like(input_ids)
        output_ids = model.generate(
                input_ids,
                max_new_tokens=128,
                temperature=temperature,
                eos_token_id=[tokenizer.eos_token_id,
                                tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                pad_token_id=tokenizer.pad_token_id,
        )[0]

        gen_ids = output_ids[input_ids.shape[-1]:]          # strip prompt
        reply   = tokenizer.decode(gen_ids, skip_special_tokens=False)
        print(reply)
        gen_tok_list.append(len(gen_ids))
# ------------------------------ REPORT ------------------------------

avg_tokens = statistics.mean(gen_tok_list)-1

print(f"Average generated tokens: {avg_tokens:.1f}")
