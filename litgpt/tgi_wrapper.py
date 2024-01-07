import requests
from llama_cpp import Llama

from litgpt.model_templates import alpaca_completion

DEFAULT_URL = "http://127.0.0.1:8000/generate"


def tgi_completion(
    prompt: str,
    url: str = DEFAULT_URL,
    max_new_tokens: int = 4096,
    top_k: int = 30,
    top_p: float = 0.9,
    temperature: float = 1.0,
    repetition_penalty: float = 1.1
):
    data = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "do_sample": True,
            "temperature": temperature,
            "seed": 42,
            "top_p": top_p,
            "top_k": top_k,
            "watermark": False
        },
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url=url, json=data, headers=headers)
    data = response.json()
    out_text = data["generated_text"].strip()
    return out_text


def alpaca_tgi_completion(messages, url: str = DEFAULT_URL):
    return alpaca_completion(
        func=tgi_completion,
        messages=messages,
        url=url
    )
