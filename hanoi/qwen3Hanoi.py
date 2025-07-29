from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt_0s = """Problem description:
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
of moves."""
prompt_icl = prompt_0s + """Here are two examples:
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
start = ["""This is the starting configuration:
A = [1, 2, 3]
B = []
C = []
This is the goal configuration:
A = []
B = []
C = [1, 2, 3]""",
"""This is the starting configuration:
A = [0, 1, 2, 3]
B = []
C = []
This is the goal configuration:
A = []
B = []
C = [0,1, 2, 3]""",
"""This is the starting configuration:
A = [0, 1, 2, 3, 4]
B = []
C = []
This is the goal configuration:
A = []
B = []
C = [0,1, 2, 3, 4]
"""]
problem = start[0] + """Give me the sequence of moves to solve the puzzle from the
starting configuration, updating the lists after each move.
Please try to use as few moves as possible, and make sure to
follow the rules listed above.  Please limit your answer to a
maximum of 15 `steps. Show minimal reasoning.
Please format your answer as below:
Step 1.  Move <N> from <src> to <tgt>.
A = []
B = []
C = []"""

prompt = prompt_0s + problem

messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
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

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# print("thinking content:", thinking_content)
print("content:", content)

