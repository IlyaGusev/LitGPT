import random
from typing import List
from dataclasses import dataclass

import torch
from sentence_transformers import  util

from utils import novel_completion, encode_prompt, parse_json_output


@dataclass
class State:
    written_paragraphs: str
    last_paragraph: str
    short_memory: str
    long_memory: str
    memory_index: torch.Tensor = None
    instruction: str = None
    next_instructions: List[str] = None

    def update_index(self, embedder):
        self.memory_index = embedder.encode(self.long_memory, convert_to_tensor=True)

    def format_long_memory(self):
        return "\n\n".join([f"{i+1}. {memory}" for i, memory in enumerate(self.long_memory)])

    def parse_long_memory(self, long_memory_str):
        memories = long_memory_str.split("\n\n")
        self.long_memory = [".".join(m.split(".")[1:]).strip() for m in memories]


class RecurrentGPT:
    def __init__(self, embedder):
        self.embedder = embedder

    def get_long_memory(self, instruction, long_memory, memory_index, top_k: int = 2):
        instruction_embedding = self.embedder.encode(instruction, convert_to_tensor=True)
        memory_scores = util.cos_sim(instruction_embedding, memory_index)[0]
        top_k_idx = torch.topk(memory_scores, k=top_k)[1]
        top_k_memory = [long_memory[idx] for idx in top_k_idx]
        input_long_term_memory = '\n'.join([f"Related Paragraphs {i+1}:" + selected_memory for i, selected_memory in enumerate(top_k_memory)])
        return input_long_term_memory

    def prepare_step_prompt(self, state: State, new_character_prob=0.1):
        state.update_index(self.embedder)
        formatted_long_memory = self.get_long_memory(state.instruction, state.long_memory, state.memory_index)

        if random.random() < new_character_prob:
            new_character_prompt = f"If it is reasonable, you can introduce a new character in the output paragrah and add it into the memory."
        else:
            new_character_prompt = ""

        assert state.instruction
        return encode_prompt(
            "prompts/step.jinja",
            short_memory=state.short_memory,
            input_paragraph=state.last_paragraph,
            input_instruction=state.instruction,
            input_long_term_memory=formatted_long_memory,
            new_character_prompt=new_character_prompt
        )

    def parse_output(self, output):
        output = parse_json_output(output)
        output = {
            "output_memory": output["updated_memory"],
            "output_paragraph": output["output_paragraph"],
            "output_instruction": [
                output["instruction_1"].strip(),
                output["instruction_2"].strip(),
                output["instruction_3"].strip(),
            ]
        }
        return output

    def step(self, state: State):
        prompt = self.prepare_step_prompt(state)
        print(prompt)
        print("@@@@@@@@@@@@@")
        response = novel_completion(prompt)
        print(response)
        print("=============")
        output = self.parse_output(response)
        state.long_memory.append(state.last_paragraph)
        state.update_index(self.embedder)
        state.written_paragraphs = state.written_paragraphs + "\n\n" + output["output_paragraph"]
        state.next_instructions = output["output_instruction"]
        state.last_paragraph = output["output_paragraph"]
        state.short_memory = output["output_memory"]
        print(state)
        return state
