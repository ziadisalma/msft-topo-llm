import os, sys
from pathlib import Path
import pandas as pd
from encoding import run_trial_and_extract


def mex(s: set) -> int:
    m = 0
    while m in s:
        m += 1
    return m

def solve_stone_piles(heaps):
    xor_sum = 0
    for h in heaps:
        xor_sum ^= h
    winning = xor_sum != 0
    move = None
    if winning:
        for i, h in enumerate(heaps):
            target = h ^ xor_sum
            if target < h:
                move = (i, h - target)
                break
    return winning, move

def solve_pin_row(n, memo={}):
    def grundy(k):
        if k in memo:
            return memo[k]
        states = set()
        # remove one pin
        if k >= 1:
            states.add(grundy(k - 1))
        # remove two adjacent pins
        for split in range(k - 1):
            left = grundy(split)
            right = grundy(k - split - 2)
            states.add(left ^ right)
        g = mex(states)
        memo[k] = g
        return g

    G = grundy(n)
    winning = G != 0
    move = None
    if winning:
        # try single-pin removal
        if n >= 1 and grundy(n - 1) == 0:
            move = ('knock_one', None)
        else:
            # try two-pin removal
            for split in range(n - 1):
                if (grundy(split) ^ grundy(n - split - 2)) == 0:
                    move = ('knock_two', split + 1)
                    break
    return G, winning, move

STONE_FEW_SHOT_EXAMPLES = [
    {
        'heaps': [3, 4, 5],
        'answer': 'True; remove 2 stones from pile 1.'
    },
    {
        'heaps': [2, 2],
        'answer': 'False; no winning move.'
    }
]

STONE_COT_EXAMPLES = [
    {
        'heaps': [3, 4, 5],
        'reasoning': (
            'Compute the bitwise XOR (nim-sum): 3 ⊕ 4 ⊕ 5 = 2. '
            'Since the result is non-zero, this is a winning position. '
            'We find a heap where reducing stones cancels the nim-sum: '
            '3 ⊕ 2 = 1, so remove 2 stones from pile 1.'
        ),
        'answer': 'True; remove 2 stones from pile 1.'
    },
    {
        'heaps': [2, 2],
        'reasoning': (
            'Compute the bitwise XOR: 2 ⊕ 2 = 0. '
            'Since the result is zero, this is a losing position.'
        ),
        'answer': 'False; no winning move.'
    }
]

PIN_FEW_SHOT_EXAMPLES = [
    {
        'n': 3,
        'answer': 'False; Grundy=0.'
    },
    {
        'n': 5,
        'answer': 'True; Grundy=3; move: knock down two pins starting at position 1.'
    }
]

PIN_COT_EXAMPLES = [
    {
        'n': 3,
        'reasoning': (
            'Compute G(0)=0, G(1)=mex{G(0)}=1, G(2)=mex{G(1), G(0)^G(0)}=2. '
            'For n=3: moves are G(2)=2 (knock one) or G(0)^G(1)=1 (knock two). '
            'Available Grundy values {2,1}, so mex{1,2}=0. Thus G(3)=0 => losing.'
        ),
        'answer': 'False; Grundy=0.'
    },
    {
        'n': 5,
        'reasoning': (
            'Compute G(4)=3, so knocking one pin yields G(4)=3 ≠ 0. '
            'Try knocking two at pos 1: splits into lengths 0 and 3 with G(0)^G(3)=0. '
            'Thus this is a winning move and G(5)=mex{…}=3.'
        ),
        'answer': 'True; Grundy=3; move: knock down two pins starting at position 1.'
    }
]

def generate_stone_piles_prompt(heaps, mode='standard'):
    rules = (
        f"There are {len(heaps)} piles of stones with sizes {', '.join(map(str, heaps))}. "
        "On each turn, a player may remove any positive number of stones from exactly one pile. "
        "The player to take the last stone wins."
    )
    if mode == 'standard':
        return rules + " Is this position winning for the player about to move? If yes, provide a winning move."

    if mode == 'few_shot':
        prompt = ""
        for ex in STONE_FEW_SHOT_EXAMPLES:
            prompt += (
                f"Example: There are {len(ex['heaps'])} piles with sizes {', '.join(map(str, ex['heaps']))}. "
                "On each turn, remove stones from one pile; last stone wins. "
                "Is this position winning? Provide your answer.\n"
                f"Answer: {ex['answer']}\n\n"
            )
        prompt += rules + " Is this position winning for the player about to move? If yes, provide a winning move."
        return prompt

    if mode == 'chain_of_thought':
        prompt = ""
        for ex in STONE_COT_EXAMPLES:
            prompt += (
                f"Example: There are {len(ex['heaps'])} piles with sizes {', '.join(map(str, ex['heaps']))}. "
                "On each turn, remove stones from one pile; last stone wins. "
                "Think step by step:\n"
                f"{ex['reasoning']}\n"
                f"Answer: {ex['answer']}\n\n"
            )
        prompt += rules + " Think step by step, then answer: Is this position winning for the player about to move? If yes, provide a winning move."
        return prompt

    if mode == 'single_token':
        return rules + " Answer only 'yes' or 'no': is this position winning for the player to move?"

    raise ValueError(f"Unknown mode: {mode}")

def generate_pin_row_prompt(n, mode='standard'):
    rules = (
        f"Consider a row of {n} pins. In a move, a player may knock down either one pin or two adjacent pins; "
        "any separate rows that form are played independently under the same rules. "
        "The player to knock down the last pin wins."
    )
    if mode == 'standard':
        return rules + " Is this position winning for the player to move? What is its Grundy value?"

    if mode == 'few_shot':
        prompt = ""
        for ex in PIN_FEW_SHOT_EXAMPLES:
            prompt += (
                f"Example: Consider a row of {ex['n']} pins under the same rules. "
                "Is this position winning? What is its Grundy value?\n"
                f"Answer: {ex['answer']}\n\n"
            )
        prompt += rules + " Is this position winning for the player to move? What is its Grundy value?"
        return prompt

    if mode == 'chain_of_thought':
        prompt = ""
        for ex in PIN_COT_EXAMPLES:
            prompt += (
                f"Example: Consider a row of {ex['n']} pins under the same rules. "
                "Think step by step:\n"
                f"{ex['reasoning']}\n"
                f"Answer: {ex['answer']}\n\n"
            )
        prompt += rules + " Think step by step, then answer: Is this position winning for the player to move? What is its Grundy value?"
        return prompt

    if mode == 'single_token':
        return rules + " Answer only 'yes' or 'no': is this position winning for the player to move?"

    raise ValueError(f"Unknown mode: {mode}")

def run_all_trials(output_csv='prompting/nim_trials.csv'):
    trials = []
    stone_tests = [(7, 4, 9), (5, 1, 4), (10, 12, 3)]
    pin_tests = [8, 5, 10]
    modes = ['standard', 'few_shot', 'chain_of_thought', 'single_token']

    # Stone Piles
    for heaps in stone_tests:
        for mode in modes:
            prompt = generate_stone_piles_prompt(heaps, mode=mode)
            winning, move = solve_stone_piles(list(heaps))
            if mode == 'single_token':
                gt = 'yes' if winning else 'no'
            else:
                if winning:
                    pile, remove = move
                    gt = f"True; remove {remove} stones from pile {pile+1}."
                else:
                    gt = "False; no winning move."
            conds = ('stone_piles', mode, ','.join(map(str, heaps)))
            trial = run_trial_and_extract(
                prompt,
                ground_truth=gt,
                conditions=conds,
                max_new_tokens=1 if mode=='single_token' else 128
            )
            trials.append(trial)

    # Pin Row
    for n in pin_tests:
        for mode in modes:
            prompt = generate_pin_row_prompt(n, mode=mode)
            G, winning, move = solve_pin_row(n)
            if mode == 'single_token':
                gt = 'yes' if winning else 'no'
            else:
                if winning:
                    if move[0] == 'knock_one':
                        mv = 'knock down one pin'
                    else:
                        mv = f"knock down two pins starting at position {move[1]}"
                    gt = f"True; Grundy={G}; move: {mv}."
                else:
                    gt = f"False; Grundy={G}."
            conds = ('pin_row', mode, f'n={n}')
            trial = run_trial_and_extract(
                prompt,
                ground_truth=gt,
                conditions=conds,
                max_new_tokens=1 if mode=='single_token' else 128
            )
            trials.append(trial)

    # Serialize results
    records = []
    for t in trials:
        records.append({
            'prompt': t.prompt,
            'response': t.response,
            'ground_truth': t.ground_truth,
            'condition_1': t.conditions[0],
            'condition_2': t.conditions[1],
            'condition_3': t.conditions[2]
        })
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    run_all_trials()
