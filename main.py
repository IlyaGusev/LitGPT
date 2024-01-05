import json

import fire
from sentence_transformers import SentenceTransformer

from litgpt.utils import DEFAULT_MODEL, encode_prompt
from litgpt.recurrentgpt import RecurrentGPT, State, gen_init_state
from litgpt.human_simulator import Human


def main(
    n_iter: int = 1,
    out_file: str = "out.jsonl",
    novel_type: str = "science fiction",
    description: str = "",
    model_name: str = DEFAULT_MODEL
):
    state = gen_init_state(
        novel_type=novel_type,
        description=description,
        model_name=model_name
    )
    embedder = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
    writer = RecurrentGPT(embedder=embedder, model_name=model_name)
    human = Human(model_name=model_name)

    with open(out_file, "w") as w:
        w.write(json.dumps(state.to_dict(), ensure_ascii=False) + "\n")
        for iter_num in range(n_iter):
            state = human.step(state)
            state = writer.step(state)
            w.write(json.dumps(state.to_dict(), ensure_ascii=False) + "\n")


if __name__ == '__main__':
    fire.Fire(main)
