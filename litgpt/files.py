import pathlib
import os

DIR_PATH = pathlib.Path(__file__).parent.resolve()
ROOT_DIR_PATH = DIR_PATH.parent.resolve()
MODELS_DIR_PATH = ROOT_DIR_PATH / "models"
PROMPTS_DIR_PATH = DIR_PATH/ "prompts"
LOCAL_MODELS_LIST = tuple(f for f in os.listdir(MODELS_DIR_PATH) if f.endswith(".gguf"))
