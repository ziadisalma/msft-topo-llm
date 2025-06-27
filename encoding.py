from dataclasses import dataclass, field
from typing import Tuple, Dict
import torch

@dataclass
class TrialData:
    prompt: str
    response: str
    ground_truth: str
    conditions: Tuple[str, str, str]
    hidden_states: Dict[int, torch.Tensor]

def run_trial_and_extract(
    prompt,
    ground_truth,
    conditions
    max_new_tokens = 1
):
    inputs = encode_prompt(prompt)
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    response = tokenizer.decode(gen_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    hidden_states = {
        layer_idx: hs_layer[0].cpu().detach()
        for layer_idx, hs_layer in enumerate(outputs.hidden_states)
    }

    return TrialData(
        prompt=prompt,
        response=response,
        ground_truth=ground_truth,
        conditions=conditions,
        hidden_states=hidden_states
    )
