from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re, statistics

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



BASE_PROMPT = (
    """Imagine a world with six rooms. From the lobby you have two choices, room 1 and room 2.
You enter room 1, at the end there’s a door that leads to room 3, and room 3 leads to room 5.
There’s a chest in room 5. You open it and there’s 10 dollars. Then you exit and start over.
This time in the lobby you choose room 2, then enter room 4, which leads to room 6. There’s
a chest with 50 dollars. You return to the lobby. Which room will you choose to make the
most money?"""
)

REVAL_PROMPT = (
    """new scenario: Now you’re dropped in room 3 and the door at its end suddenly leads to room 6, and then
you’re dropped in room 4 and the door at its end suddenly leads to room 5. you return to the
lobby. Which room will lead to more rewards? Only give room number"""
)
SYSTEM_MSG = {"role": "system", "content": "You are a concise, accurate planner."}



K = 10 
new_token_counts = []

for _ in range(K):
        convo = [SYSTEM_MSG, {"role": "user", "content": BASE_PROMPT}]


        convo.append({"role": "user", "content": REVAL_PROMPT})
        answer, n_tokens = chat(convo, temperature=0.5)
        print("\n", answer, "\n")
        new_token_counts.append(n_tokens)

print(f"Average generated tokens on reval: {statistics.mean(new_token_counts):.1f}")
