from typing import List, Dict

import requests

from tale_studio.model_settings import ModelSettings
from tale_studio.prompt_templates import format_template


DEFAULT_URL = "http://127.0.0.1:8000/generate"


def tgi_completion(
    messages: List[Dict[str, str]],
    model_settings: ModelSettings,
    url: str = DEFAULT_URL,
):
    prompt = format_template(messages, model_settings.prompt_template)
    params = vars(model_settings.generation_params)
    data = {
        "inputs": prompt,
        "parameters": {"do_sample": True, "seed": 42, "watermark": False, **params},
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url=url, json=data, headers=headers)
    data = response.json()
    out_text = data["generated_text"].strip()
    return out_text
