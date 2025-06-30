import sys
from prompting import previous_actions, cues, rules, make_fewshot_cot_prompt
from load_llama import generate_response

def main():
    prompt, expected_answer = make_fewshot_cot_prompt(
        previous_actions,
        cues,
        rules,
        num_examples=3
    )
    
    print("\n=== Prompt ===\n")
    print(prompt)
    
    reply = generate_response(prompt, max_new_tokens=1)
    
    print("\n=== Model Response (1 token) ===\n")
    print(repr(reply))
    
    print("\n=== Expected Answer ===\n")
    print(expected_answer)
