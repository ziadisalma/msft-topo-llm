import random
from load_llama import generate_response
from encoding import run_trial_and_extract

LL_COT_EXAMPLES = [
    {
        "question": 'Take the last letters of the words in "John Smith" and concatenate them.',
        "reasoning": [
            'The last letter of "John" is "n".',
            'The last letter of "Smith" is "h".',
            'Concatenating them is "nh".'
        ],
        "answer": "nh"
    },
    {
        "question": 'Take the last letters of the words in "Larry Page" and concatenate them.',
        "reasoning": [
            'The last letter of "Larry" is "y".',
            'The last letter of "Page" is "e".',
            'Concatenating them is "ye".'
        ],
        "answer": "ye"
    },
    {
        "question": 'Take the last letters of the words in "Sergey Brin" and concatenate them.',
        "reasoning": [
            'The last letter of "Sergey" is "y".',
            'The last letter of "Brin" is "n".',
            'Concatenating them is "yn".'
        ],
        "answer": "yn"
    },
    {
        "question": 'Take the last letters of the words in "Bill Gates" and concatenate them.',
        "reasoning": [
            'The last letter of "Bill" is "l".',
            'The last letter of "Gates" is "s".',
            'Concatenating them is "ls".'
        ],
        "answer": "ls"
    },
]

LL_STD_EXAMPLES = [
    {"question": ex["question"], "answer": ex["answer"]}
    for ex in LL_COT_EXAMPLES
]

CF_COT_EXAMPLES = [
    {
        "question": "A coin is heads up. Ka flips the coin. Sherrie flips the coin. Is the coin still heads up?",
        "reasoning": [
            "The coin was flipped by Ka and Sherrie.",
            "So the coin was flipped 2 times, which is an even number.",
            "The coin started heads up, so after an even number of flips, it will still be heads up."
        ],
        "answer": "yes"
    },
    {
        "question": "A coin is heads up. Jamey flips the coin. Teressa flips the coin. Is the coin still heads up?",
        "reasoning": [
            "The coin was flipped by Jamey and Teressa.",
            "So the coin was flipped 2 times, which is an even number.",
            "The coin started heads up, so after an even number of flips, it will still be heads up."
        ],
        "answer": "yes"
    },
    {
        "question": "A coin is heads up. Maybelle flips the coin. Shalonda does not flip the coin. Is the coin still heads up?",
        "reasoning": [
            "The coin was flipped by Maybelle.",
            "So the coin was flipped 1 time, which is an odd number.",
            "The coin started heads up, so after an odd number of flips, it will be tails up."
        ],
        "answer": "no"
    },
    {
        "question": "A coin is heads up. Millicent does not flip the coin. Conception flips the coin. Is the coin still heads up?",
        "reasoning": [
            "The coin was flipped by Conception.",
            "So the coin was flipped 1 time, which is an odd number.",
            "The coin started heads up, so after an odd number of flips, it will be tails up."
        ],
        "answer": "no"
    },
    {
        "question": "A coin is heads up. Sal flips the coin. Raymond does not flip the coin. Is the coin still heads up?",
        "reasoning": [
            "The coin was flipped by Sal.",
            "So the coin was flipped 1 time, which is an odd number.",
            "The coin started heads up, so after an odd number of flips, it will be tails up."
        ],
        "answer": "no"
    },
    {
        "question": "A coin is heads up. Conception flips the coin. Kristian does not flip the coin. Is the coin still heads up?",
        "reasoning": [
            "The coin was flipped by Conception.",
            "So the coin was flipped 1 time, which is an odd number.",
            "The coin started heads up, so after an odd number of flips, it will be tails up."
        ],
        "answer": "no"
    },
]

CF_STD_EXAMPLES = [
    {"question": ex["question"], "answer": ex["answer"]}
    for ex in CF_COT_EXAMPLES
]
