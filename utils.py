import json
import traceback

import torch
from openai_wrapper import openai_completion, encode_prompt, DEFAULT_MODEL


DEFAULT_SYSTEM_PROMPT = "You are a helpful and creative assistant for writing novel."


def novel_completion(
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
):
    messages=[{
        "role": "system",
        "content": "You are a helpful and creative assistant for writing novel."
    }, {
        "role": "user",
        "content": prompt,
    }]
    return openai_completion(messages, model_name=model_name)


def parse_json_output(output):
    start_index = output.find("{")
    end_index = output.rfind("}")
    text = output[start_index:end_index+1]
    text = text.strip()
    record = json.loads(text)
    return record


def novel_json_completion(
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
):
    while True:
        try:
            response = novel_completion(
                prompt=prompt,
                model_name=model_name,
                system_prompt=system_prompt
            )
            output = parse_json_output(response)
            break
        except Exception as e:
            print("Retry...")
            traceback.print_exc()
            continue
    return output


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))
