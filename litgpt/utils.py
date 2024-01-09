import json
import traceback

import torch
from litgpt.openai_wrapper import openai_completion, encode_prompt, DEFAULT_MODEL, OPENAI_MODELS
from litgpt.gguf_wrapper import gguf_completion
from litgpt.tgi_wrapper import tgi_completion


DEFAULT_SYSTEM_PROMPT = "You are a helpful and creative assistant for writing novel."


def novel_completion(
    prompt: str,
    prompt_template: str,
    model_name: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
):
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt,
        }
    ]
    if model_name == "tgi":
        return tgi_completion(messages, prompt_template=prompt_template)
    if model_name in OPENAI_MODELS:
        return openai_completion(messages, model_name=model_name)
    return gguf_completion(
        messages,
        model_name=model_name,
        prompt_template=prompt_template
    )


def parse_json_output(output):
    start_index = output.find("{")
    end_index = output.rfind("}")
    text = output[start_index:end_index + 1]
    text = text.strip()
    record = json.loads(text)
    return record


def novel_json_completion(
    prompt: str,
    model_name: str,
    prompt_template: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
):
    response = None
    while True:
        try:
            response = novel_completion(
                prompt=prompt,
                model_name=model_name,
                prompt_template=prompt_template,
                system_prompt=system_prompt
            )
            output = parse_json_output(response)
            break
        except Exception:
            if response:
                print(f"Response: {response}")
            traceback.print_exc()
            print("Retry...")
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
