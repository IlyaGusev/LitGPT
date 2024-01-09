from typing import List, Dict

from llama_cpp import Llama

from litgpt.prompt_templates import PROMPT_TEMPLATES
from litgpt.files import MODELS_DIR_PATH


class GGUFModels:
    models = dict()

    @classmethod
    def get_model(cls, model_name: str, n_gpu_layers: int = -1, n_ctx: int = 16384):
        if model_name not in cls.models:
            cls.models[model_name] = Llama(
                model_path=str(MODELS_DIR_PATH / model_name),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers
            )
        return cls.models[model_name]


def gguf_completion(
    messages: List[Dict[str, str]],
    model_name: str,
    prompt_template: str,
    top_k: int = 30,
    top_p: float = 0.95,
    temperature: float = 0.5,
    repetition_penalty: float = 1.1,
    n_ctx: int = 16384,
    n_gpu_layers: int = -1
):
    prompt = PROMPT_TEMPLATES[prompt_template](messages)
    model = GGUFModels.get_model(
        model_name,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers
    )
    tokens = model.tokenize(prompt.encode("utf-8"), special=True)
    generator = model.generate(
        tokens,
        top_k=top_k,
        top_p=top_p,
        temp=temperature,
        repeat_penalty=repetition_penalty
    )
    tokens = []
    for token in generator:
        tokens.append(token)
        if token == model.token_eos():
            break
    return model.detokenize(tokens).decode("utf-8", errors="ignore")
