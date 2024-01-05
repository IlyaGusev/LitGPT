import random
import json
from typing import List, Optional
from dataclasses import dataclass, asdict

import torch

from litgpt.utils import novel_json_completion, encode_prompt, cos_sim


@dataclass
class State:
    written_paragraphs: str
    last_paragraph: str
    prev_paragraph: str
    short_memory: str
    long_memory: str
    memory_index: Optional[torch.Tensor] = None
    instruction: Optional[str] = None
    next_instructions: List[str] = None

    def update_index(self, embedder):
        self.memory_index = embedder.encode(self.long_memory, convert_to_tensor=True)

    def format_long_memory(self):
        return "\n\n".join([f"{i+1}. {memory}" for i, memory in enumerate(self.long_memory)])

    def parse_long_memory(self, long_memory_str):
        memories = long_memory_str.split("\n\n")
        self.long_memory = [".".join(m.split(".")[1:]).strip() for m in memories]

    def to_dict(self):
        memory_index = self.memory_index
        self.memory_index = None
        result = asdict(self)
        self.memory_index = memory_index
        return result


class RecurrentGPT:
    def __init__(self, embedder, new_character_prob: float = 0.1, model_name: str = "gpt-3.5-turbo"):
        self.embedder = embedder
        self.new_character_prob = new_character_prob
        self.model_name = model_name

    def get_long_memory(self, instruction, long_memory, memory_index, top_k: int = 2):
        instruction_embedding = self.embedder.encode(instruction, convert_to_tensor=True)
        memory_scores = cos_sim(instruction_embedding, memory_index)[0]
        top_k_idx = torch.topk(memory_scores, k=top_k)[1]
        top_k_memory = [long_memory[idx] for idx in top_k_idx]
        return '\n'.join([f"Related Paragraphs {i+1}: {memory}" for i, memory in enumerate(top_k_memory)])

    def step(self, state: State):
        assert state.instruction

        state.update_index(self.embedder)
        formatted_long_memory = self.get_long_memory(
            state.instruction,
            state.long_memory,
            state.memory_index
        )
        add_new_character = random.random() < self.new_character_prob

        prompt = encode_prompt(
            "step.jinja",
            short_memory=state.short_memory,
            input_paragraph=state.last_paragraph,
            input_instruction=state.instruction,
            input_long_term_memory=formatted_long_memory,
            add_new_character=add_new_character
        )
        print("STEP PROMPT")
        print(prompt)
        print()

        output = novel_json_completion(prompt, model_name=self.model_name)
        print("STEP OUTPUT")
        print(json.dumps(output, ensure_ascii=False, indent=4))
        print("===========")

        state.prev_paragraph = state.last_paragraph
        state.long_memory.append(state.last_paragraph)
        state.update_index(self.embedder)

        output_paragraph = " ".join([p for p in output["output_paragraph"].split("\n") if p])
        output_paragraph = output_paragraph.strip()

        state.written_paragraphs = state.written_paragraphs.strip() + "\n\n" + output_paragraph
        state.next_instructions = [
            output["instruction_1"].strip(),
            output["instruction_2"].strip(),
            output["instruction_3"].strip(),
        ]
        state.last_paragraph = output_paragraph
        state.short_memory = output["updated_memory"]
        return state


def gen_init_state(description: str, novel_type: str, model_name: str):
    init_prompt = encode_prompt(
        "init.jinja",
        description=description,
        novel_type=novel_type
    )
    print("INIT PROMPT")
    print(init_prompt)
    print()
    init_info = novel_json_completion(init_prompt, model_name)
    print("INIT OUTPUT")
    print(json.dumps(init_info, ensure_ascii=False, indent=4))
    print("===========")

    first_paragraphs = '\n\n'.join([
        init_info["paragraph_1"],
        init_info["paragraph_2"],
        init_info["paragraph_3"]
    ])
    written_paragraphs = encode_prompt(
        "paragraphs.jinja",
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

    return state
