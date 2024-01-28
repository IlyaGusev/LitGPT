import json

import fire

from tale_studio.model_settings import ModelSettings, DEFAULT_EMBEDDER_NAME, DEFAULT_MODEL_NAME
from tale_studio.openai_wrapper import DEFAULT_MODEL
from tale_studio.recurrentgpt import RecurrentGPT
from tale_studio.human_simulator import Human


def main(
    n_iter: int = 1,
    out_file: str = "out.jsonl",
    novel_type: str = "science fiction",
    description: str = "",
    model_name: str = DEFAULT_MODEL_NAME,
    embedder_name: str = DEFAULT_EMBEDDER_NAME,
    prompt_template: str = "openai"
):
    model_settings = ModelSettings(
        embedder_name=DEFAULT_EMBEDDER_NAME,
        model_name=DEFAULT_MODEL,
        prompt_template=prompt_template
    )
    writer = RecurrentGPT(model_settings)
    human = Human(model_settings)
    state = writer.generate_plan(novel_type=novel_type, description=description)
    state = writer.generate_first_paragraphs(state)

    with open(out_file, "w") as w:
        w.write(json.dumps(state.to_dict(), ensure_ascii=False) + "\n")
        for iter_num in range(n_iter):
            state = human.step(state)
            state = writer.step(state)
            w.write(json.dumps(state.to_dict(), ensure_ascii=False) + "\n")


if __name__ == '__main__':
    fire.Fire(main)
