import json

import fire
from sentence_transformers import SentenceTransformer

from utils import DEFAULT_MODEL, encode_prompt
from recurrentgpt import RecurrentGPT, State, gen_init
from human_simulator import Human


def main(
    n_iter: int = 1,
    out_file: str = "out.jsonl",
    novel_type: str = "science fiction",
    description: str = "",
    model_name: str = DEFAULT_MODEL
):
    init_info = gen_init(
        novel_type=novel_type,
        description=description,
        model_name=model_name
    )
    first_paragraphs = '\n\n'.join([
        init_info["paragraph_1"],
        init_info["paragraph_2"],
        init_info["paragraph_3"]
    ])
    written_paragraphs = encode_prompt(
        "prompts/paragraphs.jinja",
        name=init_info["name"],
        outline=init_info["outline"],
        first_paragraphs=first_paragraphs
    )
    state = State(
        prev_paragraph=init_info["paragraph_2"],
        last_paragraph=init_info["paragraph_3"],
        short_memory=init_info["summary"],
        long_memory=[
            init_info["paragraph_1"],
            init_info['paragraph_2']
        ],
        written_paragraphs=written_paragraphs,
        next_instructions=[
            init_info["instruction_1"],
            init_info["instruction_2"],
            init_info["instruction_3"]
        ]
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
