import json
import traceback

import torch
from jinja2 import Template

from tale_studio.model_settings import ModelSettings
from tale_studio.files import PROMPTS_DIR_PATH
from tale_studio.openai_wrapper import openai_completion, OPENAI_MODELS, OpenAIDecodingArguments
from tale_studio.gguf_wrapper import gguf_completion
from tale_studio.tgi_wrapper import tgi_completion


DEFAULT_SYSTEM_PROMPT = "You are a helpful and creative assistant for writing novels."


def novel_completion(
    prompt: str,
    model_settings: ModelSettings,
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
    if model_settings.model_name == "tgi":
        output = tgi_completion(messages, model_settings)
    elif model_settings.model_name in OPENAI_MODELS:
        output = openai_completion(
            messages,
            decoding_args=OpenAIDecodingArguments(
                temperature=model_settings.generation_params.temperature,
                top_p=model_settings.generation_params.top_p,
            ),
            model_name=model_settings.model_name,
            api_key=model_settings.api_key
        )
    else:
        output = gguf_completion(messages, model_settings)
    output = output.replace("<|im_end|>", "")
    output = output.replace("</s>", "")
    return output


def parse_json_output(output):
    start_index = output.find("{")
    end_index = output.rfind("}")
    text = output[start_index:end_index + 1]
    text = text.strip()
    record = json.loads(text)
    return record


def novel_json_completion(
    prompt: str,
    model_settings: ModelSettings,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
):
    response = None
    while True:
        try:
            response = novel_completion(
                prompt=prompt,
                model_settings=model_settings,
                system_prompt=system_prompt,
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


def encode_prompt(template_name, **kwargs):
    template_path = PROMPTS_DIR_PATH / template_name
    with open(template_path) as f:
        template = Template(f.read())
    return template.render(**kwargs).strip() + "\n"
