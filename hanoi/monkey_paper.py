import random, itertools, textwrap

previous = ['LEFT', 'RIGHT']
cues = ['SQUARE', 'CIRCLE']
rules = ['STAY', 'SHIFT']

def build_trial(prev, cue, rule, answer = True):
    s = f"Previous action: {prev}\nCue: {cue}\nRule: {rule}\nAction?"
    if answer:
        next_act = prev if rule=='STAY' else ('RIGHT' if prev=='LEFT' else 'LEFT')
        s += f" {next_act}"
    return s

def make_fewshot_prompt(n_shots=8):
    all_conds = list(itertools.product(previous, cues, rules))
    random.shuffle(all_conds)
    shots = [build_trial(*cond) for cond in all_conds[:n_shots]]
    # choose a held-out query
    query_cond = all_conds[n_shots-1]
    
    # prompt = "### Task: Decide the next Action according to the Rule.\n" + "Here are a few worked examples with reasoning:\n" 
    prompt = "\n\n".join(shots) + "\n\n# New trial\n" + build_trial(*query_cond, answer=False)
    expected = build_trial(*query_cond).split()[-1]   # ground-truth answer
    return prompt, expected, query_cond
