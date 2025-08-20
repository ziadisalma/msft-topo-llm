from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re, statistics

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# ---------- load model & tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
)
# ---------- helper that talks to the model ----------

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

# ---------- the two user turns ----------

BASE_PROMPT = (
    """Imagine a world with six rooms. From the lobby you have two choices, room 1 and room 2.
You enter room 1, at the end there’s a door that leads to room 3, and room 3 leads to room 5.
There’s a chest in room 5. You open it and there’s 10 dollars. Then you exit and start over.
This time in the lobby you choose room 2, then enter room 4, which leads to room 6. There’s
a chest with 50 dollars. You return to the lobby. Which room will you choose to make the
most money?"""
)

REVAL_PROMPT = (
    """You enter the lobby and this time you encounter a new room, room 7. Room 7’s door leads to
room 8, and room 8 leads to room 6. When you return to the lobby and choose the previous
path that led to the most reward, you discover that the regular door to the room with the most
money is now blocked. You go back to the lobby. You will only be able to choose one path
that leads to the most money. Which room from the lobby will lead to the path where one can
make the most money? Only give room number."""
)
SYSTEM_MSG = {"role": "system", "content": "You are a concise, accurate planner."}

# ---------- run a few times ----------

K = 10  # how many repeat runs
new_token_counts = []

for _ in range(K):
        convo = [SYSTEM_MSG, {"role": "user", "content": BASE_PROMPT}]

    # reward re‑valuation turn
        convo.append({"role": "user", "content": REVAL_PROMPT})
        answer, n_tokens = chat(convo, temperature=0.5)
        print("\n", answer, "\n")
        new_token_counts.append(n_tokens)

print(f"Average generated tokens on reval: {statistics.mean(new_token_counts):.1f}")
