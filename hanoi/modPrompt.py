from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq
import torch

models = ["microsoft/phi-4", "meta-llama/Meta-Llama-3-8B-Instruct"]
model_id = models[1]

model_name = "Phi-4"
if model_id == models[1]:
    model_name = "Llama"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
prompt_ns = """Problem description:
- There are three lists labeled A, B, and C.
- There is a set of numbers distributed among those three lists.
- You can only move numbers from the rightmost end of one list to
the rightmost end of another list.
Rule #1:  You can only move a number if it is at the rightmost end
of its current list.
Rule #2:  You can only move a number to the rightmost end of a
list if it is larger than the other numbers in that list.
A move is valid if it satisfies both Rule #1 and Rule #2.
A move is invalid if it violates either Rule #1 or Rule #2.

Goal:  The goal is to end up in the configuration where all
numbers are in list C, in ascending order using minimum number
of moves.
"""

prompt_fs = prompt_ns + """Here are two examples:
Example 1:

This is the starting configuration:
A = [0, 1]
B = [2]
C = []

This is the goal configuration:
A = []
B = []
C = [0, 1, 2]

Here is the sequence of minimum number of moves to reach the goal
configuration from the starting configuration:

Move 2 from B to C.
A = [0, 1]
B = []
C = [2]

Move 1 from A to B.
A = [0]
B = [1]
C = [2]

Move 2 from C to B.
A = [0]
B = [1, 2]
C = []

Move 0 from A to C.
A = []
B = [1, 2]
C = [0]

Move 2 from B to A.
A = [2]
B = [1]
C = [0]

Move 1 from B to C.
A = [2]
B = []
C = [0, 1]

Move 2 from A to C.
A = []
B = []
C = [0, 1, 2]

Example 2:
This is the starting configuration:
A = [1]
B = [0]
C = [2]

This is the goal configuration:
A = []
B = []
C = [0, 1, 2]

Here is the sequence of minimum number of moves to reach the goal
configuration from the starting configuration:

Move 2 from C to A.
A = [1, 2]
B = [0]
C = []

Move 0 from B to C.
A = [1, 2]
B = []
C = [0]

Move 2 from A to B.
A = [1]
B = [2]
C = [0]

Move 1 from A to C.
A = []
B = [2]
C = [0, 1]

Move 2 from B to C.
A = []
B = []
C = [0, 1, 2]
"""

goal_3 = """This is the starting configuration:
A = [0, 1, 2]
B = []
C = []

This is the goal configuration:
A = []
B = []
C = [0, 1, 2]

Give me the sequence of moves to solve the puzzle from the
starting configuration, updating the lists after each move.
Please try to use as few moves as possible, and make sure to
follow the rules listed above.  Please limit your answer to a
maximum of 10 steps. Please explain each step you make.

Please format your answer as below:
Step 1.  Move <N> from <src> to <tgt>.
A = []
B = []
C = []
"""

goal_4 = """This is the starting configuration:
A = [0, 1, 2, 3]
B = []
C = []

This is the goal configuration:
A = []
B = []
C = [0, 1, 2, 3]

Give me the sequence of moves to solve the puzzle from the
starting configuration, updating the lists after each move.
Please try to use as few moves as possible, and make sure to
follow the rules listed above.  Please limit your answer to a
maximum of 10 steps. Please explain each step you make.

Please format your answer as below:
Step 1.  Move <N> from <src> to <tgt>.
A = []
B = []
C = []
"""

messages = [
    {"role": "system", "content": prompt_ns},
    {"role": "user", "content": goal_3},
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

prompt_type = "No-shot Prompting"
if messages[0]["content"] == prompt_fs:
    prompt_type = "Few-shot Prompting"

list_size = "Len 3"
if messages[1]["content"] == goal_4:
    list_size = "Len 4"

for i in range(0,10):
    print(f"Attempt {i} for {model_name} with {prompt_type} on {list_size} List:\n")
    outputs = model.generate(
        input_ids,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))

    numtokens = len(response)
    print(f"Number of tokens in your text: {numtokens}\n\n")