import random
import itertools

previous_actions = ['LEFT', 'RIGHT']
cues = ['SQUARE', 'CIRCLE']
rules = ['STAY', 'SHIFT']

def build_trial(previous_action, cue, rule, include_answer=True):
    if rule == 'STAY':
        correct = previous_action
    elif rule == 'SHIFT':
        correct = 'RIGHT' if previous_action == 'LEFT' else 'LEFT'
    else:
        raise ValueError(f"Unknown rule: {rule}")
    
    trial = (
        f"Previous action: {previous_action}\n"
        f"Cue: {cue}\n"
        f"Rule: {rule}\n"
        "Action?"
    )
    if include_answer:
        trial += f" {correct}"
    return trial, correct

def make_zero_shot_cot(previous_action, cue, rule):
    trial_text, correct = build_trial(previous_action, cue, rule, include_answer=False)
    prompt = (
        "### Task: Decide the next Action according to the Rule.\n"
        "Let's think step by step:\n\n"
        f"{trial_text}"
    )
    return prompt, correct

def build_trial_with_rationale(previous_action, cue, rule, include_answer=True):
    trial, correct = build_trial(previous_action, cue, rule, include_answer=False)
    if rule == 'STAY':
        rationale = (
            f"Reasoning: Since the rule is STAY, the next action "
            f"must be the same as the previous action ({previous_action})."
        )
    else:  # SHIFT
        opposite = 'RIGHT' if previous_action == 'LEFT' else 'LEFT'
        rationale = (
            f"Reasoning: Since the rule is SHIFT, we switch from {previous_action} "
            f"to its opposite, which is {opposite}."
        )
    # assemble example
    example = f"{trial}\n{rationale}\nAnswer? {correct}" if include_answer else f"{trial}\n{rationale}\nAnswer?"
    return example, correct

def make_fewshot_cot_prompt(previous_actions, cues, rules, num_examples=3):
    combos = list(itertools.product(previous_actions, cues, rules))
    random.shuffle(combos)
    examples = combos[:num_examples]
    query = combos[num_examples]
    
    prompt_lines = ["### Task: Decide the next Action according to the Rule.\n"]
    prompt_lines.append("Here are a few worked examples with reasoning:\n")
    # add examples with rationale + answer
    for pa, cue, rule in examples:
        ex_text, _ = build_trial_with_rationale(pa, cue, rule, include_answer=True)
        prompt_lines.append(ex_text + "\n")
    
    # now the new trial, without answer
    prompt_lines.append("### Now you try, and show your reasoning:\n")
    q_text, correct = build_trial_with_rationale(*query, include_answer=False)
    prompt_lines.append(q_text)
    
    full_prompt = "\n".join(prompt_lines)
    return full_prompt, correct
