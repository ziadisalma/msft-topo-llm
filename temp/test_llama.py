import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, REPO_ROOT)

from load_llama import generate_response

def main():
    prompt = "Hello, Llama! How are you today?"
    print(f">>> Prompt: {prompt}\n")

    response = generate_response(prompt, max_new_tokens=128)
    print(f"<<< Response: {response}\n")

if __name__ == "__main__":
    main()
