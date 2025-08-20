from transformers import AutoModelForCausalLM, AutoTokenizer
import statistics
import json
model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
	model_name,
	torch_dtype="auto",
	device_map="auto"
)

def chat(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True, # Switches between thinking and non-thinking modes. Default is True.
        
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    #thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    num_tokens = len(output_ids)
    return content, num_tokens

prompt = """ Imagine you are in a hotel with rooms denoted by letters and connected by two-way doors. Room M is connected to rooms F, Q, T, A, and B. Room F is also connected to rooms Q and T. Room Q is also connected to rooms P, T, K, and A. Room P is also connected to room A. Room T is also connected to rooms Z, K, and A. Room Z is also connected to rooms A, H, and B. Room K is also connected to room H.

        """
with open("definitions.json", "r") as f:
    definitions = json.load(f)

needed_metric = "Average shortest path"
definition = definitions[needed_metric]

convo = [
    {"role": "system", "content": "you are a graph analyst"},
    {"role": "user", "content": prompt},
    {"role": "user", "content": definition},
    {"role": "user", "content": "Return only a single number. Give the result without any additional explanation."}
]


k = 6
new_token_counts = []
for _ in range(k):
    
    answer, n_tokens = chat(convo)
    #print("thinking content:", thoughts)
    print("content:", answer)
    new_token_counts.append(n_tokens)
print(f"Average generated tokens on reval: {statistics.mean(new_token_counts):.1f}")

