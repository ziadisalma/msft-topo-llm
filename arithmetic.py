import random
import pandas as pd
import re
from models.load_llama import tokenizer

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

    # pad every token with spaces
    expression = f"( {left_expr} {op} {right_expr} )"

    # compute the numeric value
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
    innermost_paren_re = re.compile(r'\([^()]*\)')

    while (match := innermost_paren_re.search(current_expr)):
        sub_expr = match.group(0)
        result = eval(sub_expr)
        current_expr = current_expr.replace(sub_expr, str(result), 1)
        steps.append(current_expr)

    reasoning = steps[0]
    if len(steps) > 1:
        reasoning += "\n= " + "\n= ".join(steps[1:])
    return reasoning

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
            m = random.randint(*num_operands_range)
            ex_expr, ex_ans = generate_expression_and_answer(m)
            examples.append(f"{ex_expr} = {ex_ans}")
        main_m = random.randint(*num_operands_range)
        main_expr, main_ans = generate_expression_and_answer(main_m)
        prompt = "Based on the examples below, solve the final expression.\n\n"
        prompt += "\n".join(examples)
        prompt += f"\n\nNow, solve this:\n{main_expr}"
        data.append({'prompt': prompt, 'answer': main_ans})

    for _ in range(num_reasoning):
        examples = []
        for i in range(k):
            m = random.randint(*num_operands_range)
            ex_expr, ex_ans = generate_expression_and_answer(m)
            reasoning = generate_reasoning_steps(ex_expr)
            examples.append(
                f"Example {i+1}:\n"
                f"Q: What is {ex_expr}?\n"
                f"A: Let's solve it step-by-step:\n{reasoning}\nThe final answer is {ex_ans}."
            )
        main_m = random.randint(*num_operands_range)
        main_expr, main_ans = generate_expression_and_answer(main_m)
        prompt = (
            "Here are some examples with detailed reasoning. Use them to solve the final problem.\n\n"
            + "\n\n---\n\n".join(examples)
            + f"\n\n---\n\nNow, it's your turn:\nQ: What is {main_expr}?"
        )
        data.append({'prompt': prompt, 'answer': main_ans})

    return pd.DataFrame(data)


def create_varied_m_df(ms, samples_per_m):
    rows = []
    for m in ms:
        for _ in range(samples_per_m):
            expr, ans = generate_expression_and_answer(m)
            prompt = f"What is the value of the following expression?\n{expr}"

            # tokenize the FULL prompt with the Llama tokenizer
            enc = tokenizer(prompt, add_special_tokens=False)
            token_ids = enc['input_ids']
            raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)

            # normalize each token via the tokenizer's detokenization and strip whitespace
            normalized_tokens = [tokenizer.convert_tokens_to_string([tok]).strip() for tok in raw_tokens]

            # identify operand/operator positions on normalized tokens
            operand_indices = [i for i, nt in enumerate(normalized_tokens) if re.fullmatch(r"\d+", nt)]
            operator_indices = [i for i, nt in enumerate(normalized_tokens) if nt in OPERATORS]

            rows.append({
                'num_operands': m,
                'prompt': prompt,
                'answer': ans,
                'expression': expr,
                'token_ids': token_ids,
                'raw_tokens': raw_tokens,
                'normalized_tokens': normalized_tokens,
                'operand_indices': operand_indices,
                'operator_indices': operator_indices,
            })

    return pd.DataFrame(rows)

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
    mixed_df = create_varied_m_df(ms=list(range(2, N_OPERANDS+1)), samples_per_m=2)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    print(prompt_df)
    print("\nMixed-m DataFrame:\n", mixed_df)
