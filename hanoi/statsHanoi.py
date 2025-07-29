import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load model and tokenizer
model_name = "microsoft/phi-4"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with attention in 'eager' mode so output_attentions works
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,         # Optional for efficiency
    attn_implementation="eager",       # REQUIRED: disables SDPA to allow output_attentions
    output_attentions=True             # Enable attention output
)
model.eval()

# Example data
data = [
    {"text": """To solve the puzzle, we need to move all numbers to list C in ascending order using the minimum number of moves. Here's the sequence of moves:
**Step 1:** Move 2 from A to C.
- A = [0, 1]
- B = []
- C = [2]
**Step 2:** Move 1 from A to B.
- A = [0]
- B = [1]
- C = [2]
**Step 3:** Move 1 from B to C.
- A = [0]
- B = []
- C = [1, 2]
**Step 4:** Move 0 from A to B.
- A = []
- B = [0]
- C = [1, 2]
**Step 5:** Move 0 from B to C.
- A = []
- B = []
- C = [0, 1, 2]
Now all numbers are in list C in ascending order.
""", "pathway": "incorrect"},
    {"text": """To solve the puzzle and move all numbers to list C in ascending order, follow these steps:
Step 1: Move 2 from A to C.
- A = [0, 1]
- B = []
- C = [2]
Step 2: Move 1 from A to B.
- A = [0]
- B = [1]
- C = [2]
Step 3: Move 1 from B to C.
- A = [0]
- B = []
- C = [1, 2]
Step 4: Move 0 from A to B.
- A = []
- B = [0]
- C = [1, 2]
Step 5: Move 0 from B to C.
- A = []
- B = []
- C = [0, 1, 2]
The numbers are now in list C in ascending order.
""", "pathway": "incorrect"},
    {"text": """To solve the puzzle, we need to move all numbers to list C in ascending order using the minimum number of moves. Here is the sequence of moves:
**Step 1:** Move 2 from A to C.  
- A = [0, 1]  
- B = []  
- C = [2]  
**Step 2:** Move 1 from A to B.  
- A = [0]  
- B = [1]  
- C = [2]  
**Step 3:** Move 1 from B to C.  
- A = [0]  
- B = []  
- C = [1, 2]  
**Step 4:** Move 0 from A to B.  
- A = []  
- B = [0]  
- C = [1, 2]  
**Step 5:** Move 0 from B to C.  
- A = []  
- B = []  
- C = [0, 1, 2]  
Now all numbers are in list C in ascending order.
""", "pathway": "incorrect"},
    {"text": """To solve the puzzle, we need to move the numbers from lists A and B to list C in ascending order using the minimum number of moves. Here's the sequence of moves:
**Step 1.** Move 2 from A to C.  
A = [0, 1]  
B = []  
C = [2]  
**Step 2.** Move 1 from A to B.  
A = [0]  
B = [1]  
C = [2]  
**Step 3.** Move 1 from B to C.  
A = [0]  
B = []  
C = [1, 2]  
**Step 4.** Move 0 from A to B.  
A = []  
B = [0]  
C = [1, 2]  
**Step 5.** Move 0 from B to C.  
A = []  
B = []  
C = [0, 1, 2]  
Now all numbers are in list C in ascending order.
""", "pathway": "incorrect"},
    {"text": """To solve the puzzle and move all numbers to list C in ascending order using the minimum number of moves, follow these steps:
**Initial Configuration:**
- A = [0, 1, 2]
- B = []
- C = []
**Step-by-step Moves:**
**Step 1:** Move 2 from A to C.
- A = [0, 1]
- B = []
- C = [2]
**Step 2:** Move 1 from A to B.
- A = [0]
- B = [1]
- C = [2]
**Step 3:** Move 1 from B to C.
- A = [0]
- B = []
- C = [1, 2]
**Step 4:** Move 0 from A to B.
- A = []
- B = [0]
- C = [1, 2]
**Step 5:** Move 0 from B to C.
- A = []
- B = []
- C = [0, 1, 2]
**Final Configuration:**
- A = []
- B = []
- C = [0, 1, 2]
This sequence of moves ensures that all numbers are in list C in ascending
""", "pathway": "incorrect"},
    {"text": """To solve the puzzle, we need to move all numbers from lists A and B to list C in ascending order. Here's the sequence of moves:
**Step 1:** Move 2 from A to C.
- A = [0, 1]
- B = []
- C = [2]
**Step 2:** Move 1 from A to B.
- A = [0]
- B = [1]
- C = [2]
**Step 3:** Move 1 from B to C.
- A = [0]
- B = []
- C = [1, 2]
**Step 4:** Move 0 from A to B.
- A = []
- B = [0]
- C = [1, 2]
**Step 5:** Move 0 from B to C.
- A = []
- B = []
- C = [0, 1, 2]
Now, all numbers are in list C in ascending order.
""", "pathway": "incorrect"},
    {"text": """To solve the puzzle and move all numbers to list C in ascending order using the minimum number of moves, follow these steps:
**Starting Configuration:**
- A = [0, 1, 2]
- B = []
- C = []
**Step 1:**
- Move 2 from A to C.
- A = [0, 1]
- B = []
- C = [2]
**Step 2:**
- Move 1 from A to B.
- A = [0]
- B = [1]
- C = [2]
**Step 3:**
- Move 1 from B to C.
- A = [0]
- B = []
- C = [1, 2]
**Step 4:**
- Move 0 from A to B.
- A = []
- B = [0]
- C = [1, 2]
**Step 5:**
- Move 0 from B to C.
- A = []
- B = []
- C = [0, 1, 2]
**Goal Configuration:**
- A = []
- B = []
- C = [0, 1, 2]
This sequence of moves ensures all numbers are in list C
""", "pathway": "incorrect"},
    {"text": """To solve the puzzle, we need to move all numbers from lists A and B to list C in ascending order using the minimum number of moves. Here's the sequence of moves:
**Step 1:** Move 2 from A to C.
- A = [0, 1]
- B = []
- C = [2]
**Step 2:** Move 1 from A to B.
- A = [0]
- B = [1]
- C = [2]
**Step 3:** Move 1 from B to C.
- A = [0]
- B = []
- C = [1, 2]
**Step 4:** Move 0 from A to B.
- A = []
- B = [0]
- C = [1, 2]
**Step 5:** Move 0 from B to C.
- A = []
- B = []
- C = [0, 1, 2]
Now, all numbers are in list C in ascending order.
""", "pathway": "incorrect"},
    {"text": """To solve the puzzle and move all numbers to list C in ascending order using the minimum number of moves, follow these steps:
Step 1. Move 2 from A to C.
- A = [0, 1]
- B = []
- C = [2]
Step 2. Move 1 from A to B.
- A = [0]
- B = [1]
- C = [2]
Step 3. Move 1 from B to C.
- A = [0]
- B = []
- C = [1, 2]
Step 4. Move 0 from A to B.
- A = []
- B = [0]
- C = [1, 2]
Step 5. Move 0 from B to C.
- A = []
- B = []
- C = [0, 1, 2]
Now, all numbers are in list C in ascending order.
""", "pathway": "incorrect"},
    {"text": """To solve the puzzle, we need to move all numbers from lists A and B to list C in ascending order. Here's the sequence of moves:
**Starting Configuration:**
- A = [0, 1, 2]
- B = []
- C = []
**Step-by-Step Moves:**
**Step 1.** Move 2 from A to C.
- A = [0, 1]
- B = []
- C = [2]
**Step 2.** Move 1 from A to C.
- A = [0]
- B = []
- C = [2, 1]
**Step 3.** Move 2 from C to B.
- A = [0]
- B = [2]
- C = [1]
**Step 4.** Move 1 from C to A.
- A = [0, 1]
- B = [2]
- C = []
**Step 5.** Move 1 from A to C.
- A = [0]
- B = [2]
- C = [1]
**Step 6.** Move 0 from A to C.
- A = []
- B = [2]
- C = [1,
""", "pathway": "incorrect"},
    {"text": """To solve the puzzle, we need to move the numbers from lists A and B to list C, following the rules. Here's the sequence of moves:
Step 1. Move 2 from A to C.
- A = [0, 1]
- B = []
- C = [2]
Step 2. Move 1 from A to B.
- A = [0]
- B = [1]
- C = [2]
Step 3. Move 2 from C to B.
- A = [0]
- B = [1, 2]
- C = []
Step 4. Move 0 from A to C.
- A = []
- B = [1, 2]
- C = [0]
Step 5. Move 2 from B to A.
- A = [2]
- B = [1]
- C = [0]
Step 6. Move 1 from B to C.
- A = [2]
- B = []
- C = [0, 1]
Step 7. Move 2 from A to C.
- A = []
- B = []
- C = [0, 1, 2]
This sequence uses 7 moves to reach the
""", "pathway": "correct"},
    {"text": """To solve the puzzle, we need to move the numbers from lists A and B to list C in ascending order, following the given rules. Here's the sequence of moves:
Step 1. Move 2 from A to C.
- A = [0, 1]
- B = []
- C = [2]
Step 2. Move 1 from A to B.
- A = [0]
- B = [1]
- C = [2]
Step 3. Move 2 from C to B.
- A = [0]
- B = [1, 2]
- C = []
Step 4. Move 0 from A to C.
- A = []
- B = [1, 2]
- C = [0]
Step 5. Move 2 from B to A.
- A = [2]
- B = [1]
- C = [0]
Step 6. Move 1 from B to C.
- A = [2]
- B = []
- C = [0, 1]
Step 7. Move 2 from A to C.
- A = []
- B = []
- C = [0, 1, 2]
This sequence of moves successfully
""", "pathway": "correct"},
    {"text": """Here is the sequence of moves to solve the puzzle:
Step 1. Move 2 from A to C.
A = [0, 1]
B = []
C = [2]
Step 2. Move 1 from A to B.
A = [0]
B = [1]
C = [2]
Step 3. Move 2 from C to B.
A = [0]
B = [1, 2]
C = []
Step 4. Move 0 from A to C.
A = []
B = [1, 2]
C = [0]
Step 5. Move 2 from B to A.
A = [2]
B = [1]
C = [0]
Step 6. Move 1 from B to C.
A = [2]
B = []
C = [0, 1]
Step 7. Move 2 from A to C.
A = []
B = []
C = [0, 1, 2]
""", "pathway": "correct"},
    {"text": """Here is the sequence of moves to solve the puzzle:
Step 1. Move 2 from A to C.
A = [0, 1]
B = []
C = [2]
Step 2. Move 1 from A to B.
A = [0]
B = [1]
C = [2]
Step 3. Move 2 from C to B.
A = [0]
B = [1, 2]
C = []
Step 4. Move 0 from A to C.
A = []
B = [1, 2]
C = [0]
Step 5. Move 2 from B to A.
A = [2]
B = [1]
C = [0]
Step 6. Move 1 from B to C.
A = [2]
B = []
C = [0, 1]
Step 7. Move 2 from A to C.
A = []
B = []
C = [0, 1, 2]
""", "pathway": "correct"},
    {"text": """Here is the sequence of moves to reach the goal configuration:
Step 1. Move 2 from A to C.
A = [0, 1]
B = []
C = [2]
Step 2. Move 1 from A to B.
A = [0]
B = [1]
C = [2]
Step 3. Move 2 from C to B.
A = [0]
B = [1, 2]
C = []
Step 4. Move 0 from A to C.
A = []
B = [1, 2]
C = [0]
Step 5. Move 2 from B to A.
A = [2]
B = [1]
C = [0]
Step 6. Move 1 from B to C.
A = [2]
B = []
C = [0, 1]
Step 7. Move 2 from A to C.
A = []
B = []
C = [0, 1, 2]
Final configuration:
A = []
B = []
C = [0, 1, 2]
""", "pathway": "correct"},
    {"text": """To solve the puzzle with the given starting configuration and reach the goal configuration, follow these steps:
Step 1. Move 2 from A to C.
- A = [0, 1]
- B = []
- C = [2]
Step 2. Move 1 from A to B.
- A = [0]
- B = [1]
- C = [2]
Step 3. Move 2 from C to B.
- A = [0]
- B = [1, 2]
- C = []
Step 4. Move 0 from A to C.
- A = []
- B = [1, 2]
- C = [0]
Step 5. Move 2 from B to A.
- A = [2]
- B = [1]
- C = [0]
Step 6. Move 1 from B to C.
- A = [2]
- B = []
- C = [0, 1]
Step 7. Move 2 from A to C.
- A = []
- B = []
- C = [0, 1, 2]
Now, all numbers are in list C in ascending order, and the goal configuration is achieved.
""", "pathway": "correct"},
    {"text": """To solve the puzzle from the starting configuration to the goal configuration, follow these steps:
Step 1. Move 2 from A to C.
- A = [0, 1]
- B = []
- C = [2]
Step 2. Move 1 from A to B.
- A = [0]
- B = [1]
- C = [2]
Step 3. Move 2 from C to B.
- A = [0]
- B = [1, 2]
- C = []
Step 4. Move 0 from A to C.
- A = []
- B = [1, 2]
- C = [0]
Step 5. Move 2 from B to A.
- A = [2]
- B = [1]
- C = [0]
Step 6. Move 1 from B to C.
- A = [2]
- B = []
- C = [0, 1]
Step 7. Move 2 from A to C.
- A = []
- B = []
- C = [0, 1, 2]
This sequence of moves results in the goal configuration using the minimum number of moves.
""", "pathway": "correct"},
    {"text": """Here's the sequence of moves to solve the puzzle:
Step 1. Move 2 from A to C.
A = [0, 1]
B = []
C = [2]
Step 2. Move 1 from A to B.
A = [0]
B = [1]
C = [2]
Step 3. Move 2 from C to B.
A = [0]
B = [1, 2]
C = []
Step 4. Move 0 from A to C.
A = []
B = [1, 2]
C = [0]
Step 5. Move 2 from B to A.
A = [2]
B = [1]
C = [0]
Step 6. Move 1 from B to C.
A = [2]
B = []
C = [0, 1]
Step 7. Move 2 from A to C.
A = []
B = []
C = [0, 1, 2]
Final configuration:
A = []
B = []
C = [0, 1, 2]
""", "pathway": "correct"},
    {"text": """To solve the puzzle from the starting configuration to the goal configuration, follow these steps:
Step 1. Move 2 from A to C.
- A = [0, 1]
- B = []
- C = [2]
Step 2. Move 1 from A to B.
- A = [0]
- B = [1]
- C = [2]
Step 3. Move 2 from C to B.
- A = [0]
- B = [1, 2]
- C = []
Step 4. Move 0 from A to C.
- A = []
- B = [1, 2]
- C = [0]
Step 5. Move 2 from B to A.
- A = [2]
- B = [1]
- C = [0]
Step 6. Move 1 from B to C.
- A = [2]
- B = []
- C = [0, 1]
Step 7. Move 2 from A to C.
- A = []
- B = []
- C = [0, 1, 2]
This sequence of moves achieves the goal configuration in 7 steps.
""", "pathway": "correct"},
    {"text": """To solve the puzzle from the starting configuration to the goal configuration, follow these steps:
**Starting Configuration:**
- A = [0, 1, 2]
- B = []
- C = []
**Goal Configuration:**
- A = []
- B = []
- C = [0, 1, 2]
**Sequence of Moves:**
Step 1. Move 2 from A to C.
- A = [0, 1]
- B = []
- C = [2]
Step 2. Move 1 from A to B.
- A = [0]
- B = [1]
- C = [2]
Step 3. Move 2 from C to B.
- A = [0]
- B = [1, 2]
- C = []
Step 4. Move 0 from A to C.
- A = []
- B = [1, 2]
- C = [0]
Step 5. Move 2 from B to A.
- A = [2]
- B = [1]
- C = [0]
Step 6. Move 1 from B to C.
- A = [2]
- B = []
- C = [0, 1
""", "pathway": "correct"},
]

results = []

""" # OLD CODE FOR LOOP
for example in data:
    inputs = tokenizer(example["text"], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions  # list of (1, num_heads, seq_len, seq_len)

    if attentions is None:
        raise RuntimeError("Attention outputs are None. Ensure 'attn_implementation=\"eager\"' is set.")

    for layer_idx, attn_tensor in enumerate(attentions):
        # Average over heads
        attn_avg = attn_tensor.mean(dim=1)  # (1, seq_len, seq_len)
        avg_score = attn_avg.mean().item()  # overall average attention

        results.append({
            "avg_attention": avg_score,
            "pathway": example["pathway"],
            "layer": layer_idx
        })
"""

for example in data:
    inputs = tokenizer(example["text"], return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    target_index = seq_len - 1  # Final token index

    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions  # list of (1, num_heads, seq_len, seq_len)

    for layer_idx, attn_tensor in enumerate(attentions):
        # Shape: (1, num_heads, seq_len, seq_len)
        # Get attention TO the final token (i.e., all queries → final key)
        attn_to_final = attn_tensor[:, :, :, target_index]  # (1, num_heads, seq_len)

        # Average across all heads and queries (query dimension is axis 2)
        attn_score = attn_to_final.mean().item()

        results.append({
            "avg_attention": attn_score,
            "pathway": example["pathway"],
            "layer": layer_idx
        })


# Create DataFrame
df = pd.DataFrame(results)

# Avoid logit issues
epsilon = 1e-5
df['avg_attention'] = df['avg_attention'].clip(epsilon, 1 - epsilon)
df['logit_attention'] = np.log(df['avg_attention'] / (1 - df['avg_attention']))

"""
# Mixed-effects model
model = smf.mixedlm("logit_attention ~ pathway", df, groups=df["layer"])
result = model.fit()
print(result.summary())
"""

# Fit the model
model = smf.mixedlm("logit_attention ~ pathway", df, groups=df["layer"])
result = model.fit()

# Extract fixed effects
fe_params = result.fe_params
fe_se = result.bse_fe  # standard errors of fixed effects

# Compute t-statistics manually
t_stats = fe_params / fe_se

# Approximate degrees of freedom using the number of groups (layers) minus 1
n_groups = df['layer'].nunique()
df_approx = n_groups - 1  # conservative estimate

# Combine everything into a summary table
summary_df = pd.DataFrame({
    'Estimate': fe_params,
    'Std.Err': fe_se,
    't value': t_stats,
    'df (approx)': df_approx
})

summary_df["p-value"] = 2 * (1 - stats.t.cdf(abs(t_stats), df=df_approx))

print("\nMixed Effects Model Summary:")
print(summary_df)
