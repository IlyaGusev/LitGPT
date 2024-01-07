from llama_cpp import Llama
from litgpt.model_templates import alpaca_completion

MODEL2PATH = {
    "neural-chat-7b-v3-3": "models/neural-chat-7b-v3-3.Q5_K_M.gguf"
}


class GGUFModels:
    models = dict()

    @classmethod
    def get_model(cls, model_name: str):
        if model_name not in cls.models:
            cls.models[model_name] = Llama(
                model_path=MODEL2PATH[model_name],
                n_parts=1,
                n_ctx=16384
            )
        return cls.models[model_name]


def gguf_completion(
    prompt: str,
    model_name: str,
    top_k: int = 30,
    top_p: float = 0.95,
    temperature: float = 0.5,
    repetition_penalty: float = 1.1,
):
    model = GGUFModels.get_model(model_name)
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


def alpaca_gguf_completion(messages, model_name: str):
    return alpaca_completion(
        func=gguf_completion,
        messages=messages,
        model_name=model_name
    )
