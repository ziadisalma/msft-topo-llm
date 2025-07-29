from transformers import AutoModelForCausalLM, AutoTokenizer

models = ["Qwen/Qwen2.5-7B-Instruct-1M", "Qwen/Qwen3-8B"] 

model_name = models[0]

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = """Given three lists, A, B, and C, where only the largest element in each list can be moved to another list, and an element can only be added to a list if it is larger than all other elements in the list, determine if all elements of list A can be moved to list C in 7 moves?
Answer "Yes" or "No" and provide minimal reasoning.
The lists are: A = [1,2,3], B = [] and C = []."""

messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": prompt + "Answer:"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

for i in range(0,10):
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=768
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

    num_response_tokens = len(generated_ids[0])
    print(f"Number of tokens in response: {num_response_tokens}")

