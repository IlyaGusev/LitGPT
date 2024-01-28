from dataclasses import dataclass


DEFAULT_MODEL_NAME = "gpt-3.5-turbo-16k"
DEFAULT_EMBEDDER_NAME = "embaas/sentence-transformers-multilingual-e5-base"


@dataclass
class GenerationParams:
    temperature: float = 0.6
    repetition_penalty: float = 1.2
    top_p: float = 0.9
    top_k: int = 30
    max_new_tokens: int = 4096


@dataclass
class ModelSettings:
    model_name: str = DEFAULT_MODEL_NAME
    embedder_name: str = DEFAULT_EMBEDDER_NAME
    prompt_template: str = "openai"
    api_key: str = ""
    generation_params: GenerationParams = GenerationParams()
