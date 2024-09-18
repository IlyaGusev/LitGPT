from dataclasses import dataclass, field

from tale_studio.files import LOCAL_MODELS_LIST, SAVES_DIR_PATH

DEFAULT_MODEL_NAME = LOCAL_MODELS_LIST[0] if LOCAL_MODELS_LIST else ""
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
    prompt_template: str = "chatml"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    generation_params: GenerationParams = field(default_factory=GenerationParams)
    n_ctx: int = 16384
    n_gpu_layers: int = -1
