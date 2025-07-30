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
    "Imagine a world with 4 rooms. From the lobby you have two choices, room 1 and room 2.\n"
    "You enter room 1; at the end there’s a door that leads to room 3.\n"
    "There’s a chest in room 3 with 10 dollars. You exit and return to the lobby.\n"
    "This time you choose room 2, which leads to room 4 with a chest of 50 dollars.\n"
    "Back in the lobby: Which room will you choose to make the most money?\n"
)

REVAL_PROMPT = (
    """You enter the lobby and this time you encounter a new room, room 5. Room 5’s door leads to room 4.When you return to the lobby and choose the previous path that led to the most reward, you discover th>
You go back to the lobby. You will only be able to choose one path that leads to the most money. 
Which room from the lobby will lead to the path where one can make the most money? Use minimal reasoning to get to the answer quickly."""
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
