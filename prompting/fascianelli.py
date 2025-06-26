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

def make_fewshot_prompt(previous_actions, cues, rules, num_few_shot=3):
    combos = list(itertools.product(previous_actions, cues, rules))
    random.shuffle(combos)
    
    few_shot = combos[:num_few_shot]
    query_tuple = combos[num_few_shot]
    
    prompt_lines = ["### Task: Decide the next Action according to the Rule.\n"]
    for pa, cue, rule in few_shot:
        trial_text, _ = build_trial(pa, cue, rule, include_answer=True)
        prompt_lines.append(trial_text + "\n")
    
    prompt_lines.append("# New trial\n")
    query_text, correct_answer = build_trial(*query_tuple, include_answer=False)
    prompt_lines.append(query_text)
    
    prompt = "\n".join(prompt_lines)
    return prompt, correct_answer, query_tuple
