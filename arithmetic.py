import random
import pandas as pd
import re

OPERATORS = ['+', '-', '*', '//']
OPERAND_RANGE = (1, 100)

def generate_expression_and_answer(num_operands):
    if num_operands == 1:
        operand = random.randint(*OPERAND_RANGE)
        return str(operand), operand

    split_point = random.randint(1, num_operands - 1)
    left_expr, left_val = generate_expression_and_answer(split_point)
    right_expr, right_val = generate_expression_and_answer(num_operands - split_point)

    op = random.choice(OPERATORS)

    if op == '//' and right_val == 0:
        op = random.choice(['+', '-'])

    expression = f"({left_expr} {op} {right_expr})"
    
    if op == '+':
        value = left_val + right_val
    elif op == '-':
        value = left_val - right_val
    elif op == '*':
        value = left_val * right_val
    else:  # op == '//'
        value = left_val // right_val
        
    return expression, value

def generate_reasoning_steps(expression):
    steps = [expression]
    current_expr = expression
    
    # Regex to find the innermost parenthesized expression
    innermost_paren_re = re.compile(r'\([^()]*\)')

    while (match := innermost_paren_re.search(current_expr)):
        sub_expr_with_parens = match.group(0)
        result = eval(sub_expr_with_parens)
        
        current_expr = current_expr.replace(sub_expr_with_parens, str(result), 1)
        steps.append(current_expr)

    reasoning_str = steps[0]
    if len(steps) > 1:
        reasoning_str += "\n= " + "\n= ".join(steps[1:])
        
    return reasoning_str

def create_prompts(num_zero_shot, num_few_shot, num_reasoning, k, n):
    data = []
    num_operands_range = (3, n + 1)

    for _ in range(num_zero_shot):
        expr, ans = generate_expression_and_answer(n)
        prompt = f"What is the value of the following expression?\n{expr}"
        data.append({'prompt': prompt, 'answer': ans})
        
    for _ in range(num_few_shot):
        examples = []
        for _ in range(k):
            n_ops = random.randint(*num_operands_range)
            expr, ans = generate_expression_and_answer(n_ops)
            examples.append(f"{expr} = {ans}")
        
        main_n_ops = random.randint(*num_operands_range)
        main_expr, main_ans = generate_expression_and_answer(main_n_ops)
        
        prompt = "Based on the examples below, solve the final expression.\n\n"
        prompt += "\n".join(examples)
        prompt += f"\n\nNow, solve this:\n{main_expr}"
        data.append({'prompt': prompt, 'answer': main_ans})
        
    for _ in range(num_reasoning):
        examples = []
        for i in range(k):
            n_ops = random.randint(*num_operands_range)
            expr, ans = generate_expression_and_answer(n_ops)
            reasoning = generate_reasoning_steps(expr)
            
            example_str = f"Example {i+1}:\n"
            example_str += f"Q: What is {expr}?\n"
            example_str += f"A: Let's solve it step-by-step:\n{reasoning}\nThe final answer is {ans}."
            examples.append(example_str)
            
        main_n_ops = random.randint(*num_operands_range)
        main_expr, main_ans = generate_expression_and_answer(main_n_ops)
    
        prompt = "Here are some examples with detailed reasoning. Use them to solve the final problem.\n\n"
        prompt += "\n\n---\n\n".join(examples)
        prompt += f"\n\n---\n\nNow, it's your turn:\nQ: What is {main_expr}?"
        data.append({'prompt': prompt, 'answer': main_ans})
        
    return pd.DataFrame(data)


if __name__ == '__main__':
    NUM_ZERO_SHOT = 1
    NUM_FEW_SHOT = 1
    NUM_REASONING = 1
    K_EXAMPLES = 2
    N_OPERANDS = 4

    prompt_df = create_prompts(
        num_zero_shot=NUM_ZERO_SHOT,
        num_few_shot=NUM_FEW_SHOT,
        num_reasoning=NUM_REASONING,
        k=K_EXAMPLES,
        n=N_OPERANDS
    )
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    print(prompt_df)

    # Example of accessing a single prompt and its answer
    if not prompt_df.empty:
        print("\n\n--- Example of a single prompt ---")
        print(prompt_df.iloc[0]['prompt'])
        print(f"\nCorrect Answer: {prompt_df.iloc[0]['answer']}")

