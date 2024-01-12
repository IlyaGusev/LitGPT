import json

import fire

from tale_studio.embedders import DEFAULT_EMBEDDER_NAME
from tale_studio.openai_wrapper import DEFAULT_MODEL
from tale_studio.recurrentgpt import RecurrentGPT
from tale_studio.human_simulator import Human


def main(
    n_iter: int = 1,
    out_file: str = "out.jsonl",
    novel_type: str = "science fiction",
    description: str = "",
    model_name: str = DEFAULT_MODEL,
    embedder_name: str = DEFAULT_EMBEDDER_NAME,
    prompt_template: str = "openai"
):
    writer = RecurrentGPT(
        embedder_name=embedder_name,
        model_name=model_name,
        prompt_template=prompt_template
    )
    human = Human(
        model_name=model_name,
        prompt_template=prompt_template
    )
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
