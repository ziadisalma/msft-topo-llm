from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq
import torch
models = ["microsoft/phi-4", "meta-llama/Meta-Llama-3-8B-Instruct", "microsoft/Phi-4-reasoning"]
model_id = models[2]
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
prompt = """Given three lists, A, B, and C, where only the largest element in each list can be moved to another list, and an element can only be added to a list if it is larger than all other elements in the list, determine if all elements of list A can be moved to list C in 7 moves?
Only answer "Yes" or "No".
The lists are: A = [1,2,3], B = [] and C = []."""

messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": prompt + "Answer:"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
for i in range(0,10):
    outputs = model.generate(
        input_ids,
        max_new_tokens=4096,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))
    numtokens = len(response)
    print(f"Number of tokens in your text: {numtokens}\n\n")

