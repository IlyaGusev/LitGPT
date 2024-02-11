import copy
from typing import List, Dict

from llama_cpp import Llama

from tale_studio.model_settings import ModelSettings
from tale_studio.prompt_templates import format_template
from tale_studio.files import MODELS_DIR_PATH


class GGUFModels:
    models = dict()

    @classmethod
    def get_model(cls, model_name: str, n_gpu_layers: int = -1, n_ctx: int = 16384):
        if model_name not in cls.models:
            cls.models[model_name] = Llama(
                model_path=str(MODELS_DIR_PATH / model_name),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
            )
        return cls.models[model_name]


def gguf_completion(
    messages: List[Dict[str, str]],
    model_settings: ModelSettings,
    n_ctx: int = 16384,
    n_gpu_layers: int = -1,
):
    prompt = format_template(messages, model_settings.prompt_template)
    model = GGUFModels.get_model(
        model_settings.model_name, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers
    )
    tokens = model.tokenize(prompt.encode("utf-8"), special=True)

    params = copy.deepcopy(vars(model_settings.generation_params))
    params["temp"] = params.pop("temperature")
    params["repeat_penalty"] = params.pop("repetition_penalty")
    params.pop("max_new_tokens")

    generator = model.generate(tokens, **params)
    tokens = []
    for token in generator:
        tokens.append(token)
        if token == model.token_eos():
            break
    return model.detokenize(tokens).decode("utf-8", errors="ignore")
