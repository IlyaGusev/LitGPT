import random
import json
from typing import List, Optional
from dataclasses import dataclass, asdict, field

import torch
from sentence_transformers import SentenceTransformer

from litgpt.utils import novel_json_completion, encode_prompt, cos_sim


@dataclass
class State:
    name: str
    synopsis: str
    plan: str = ""
    novel_type: str = ""
    description: str = ""
    paragraphs: List[str] = field(default_factory=lambda: list())
    short_memory: str = ""
    memory_index: Optional[torch.Tensor] = None
    instruction: str = ""
    next_instructions: List[str] = field(default_factory=lambda: list())

    @property
    def long_memory(self):
        return self.paragraphs[:-1]

    def update_index(self, embedder, passage_prefix):
        long_memory = [passage_prefix + p for p in self.long_memory]
        self.memory_index = embedder.encode(long_memory, convert_to_tensor=True)

    def to_dict(self):
        memory_index = self.memory_index
        self.memory_index = None
        result = asdict(self)
        self.memory_index = memory_index
        return result


class EmbeddersStorage:
    embedders = dict()

    @classmethod
    def get_embedder(cls, embedder_name: str):
        if embedder_name not in cls.embedders:
            cls.embedders[embedder_name] = SentenceTransformer(embedder_name)
        return cls.embedders[embedder_name]


class RecurrentGPT:
    def __init__(self, model_name: str, embedder_name: str, prompt_template: str):
        self.model_name = model_name
        self.embedder_name = embedder_name
        self.prompt_template = prompt_template
        self.embedder = EmbeddersStorage.get_embedder(embedder_name)
        self.query_prefix = "query: "
        self.passage_prefix = "passage: "

    def get_relevant_long_memory(self, instruction, long_memory, memory_index, top_k: int = 2):
        instruction_embedding = self.embedder.encode(self.query_prefix + instruction, convert_to_tensor=True)
        memory_scores = cos_sim(instruction_embedding, memory_index)[0]
        top_k_idx = torch.topk(memory_scores, k=top_k)[1]
        top_k_memory = [long_memory[idx] for idx in top_k_idx]
        return '\n'.join([f"Related Paragraphs {i+1}: {memory}" for i, memory in enumerate(top_k_memory)])

    def step(self, state: State):
        assert state.instruction

        state.update_index(self.embedder, self.passage_prefix)
        formatted_long_memory = self.get_relevant_long_memory(
            state.instruction,
            state.long_memory,
            state.memory_index
        )

        prompt = encode_prompt(
            "step.jinja",
            plan=state.plan,
            short_memory=state.short_memory,
            input_paragraph=state.paragraphs[-1],
            num_paragraphs=len(state.paragraphs),
            input_instruction=state.instruction,
            input_long_term_memory=formatted_long_memory,
        )
        print("STEP PROMPT")
        print(prompt)
        print()

        output = novel_json_completion(prompt, self.model_name, self.prompt_template)
        print("STEP OUTPUT")
        print(json.dumps(output, ensure_ascii=False, indent=4))
        print("===========")

        output_paragraph = " ".join([p for p in output["output_paragraph"].split("\n") if p])
        output_paragraph = output_paragraph.strip()
        state.paragraphs.append(output_paragraph)
        state.update_index(self.embedder, self.passage_prefix)

        state.next_instructions = [
            output["instruction_1"].strip(),
            output["instruction_2"].strip(),
            output["instruction_3"].strip(),
        ]
        state.short_memory = output["updated_memory"]
        return state

    def generate_plan(
        self,
        description: str,
        novel_type: str,
    ):
        plan_prompt = encode_prompt(
            "plan.jinja",
            description=description,
            novel_type=novel_type
        )
        print("PLAN PROMPT")
        print(plan_prompt)
        print()
        plan_info = novel_json_completion(plan_prompt, self.model_name, self.prompt_template)
        print("PLAN OUTPUT")
        print(json.dumps(plan_info, ensure_ascii=False, indent=4))
        print("===========")

        return State(
            name=plan_info["name"],
            synopsis=plan_info["synopsis"],
            plan="\n".join(plan_info["chapter_summaries"]),
            novel_type=novel_type,
            description=description
        )

    def generate_first_paragraphs(
        self,
        state: State
    ):
        init_prompt = encode_prompt(
            "init.jinja",
            novel_type=state.novel_type,
            plan=state.plan,
            name=state.name,
            synopsis=state.synopsis,
        )
        print("INIT PROMPT")
        print(init_prompt)
        print()
        init_info = novel_json_completion(init_prompt, self.model_name, self.prompt_template)
        print("INIT OUTPUT")
        print(json.dumps(init_info, ensure_ascii=False, indent=4))
        print("===========")

        state.short_memory = init_info["summary"]
        state.paragraphs = [
            init_info["paragraph_1"],
            init_info["paragraph_2"],
            init_info["paragraph_3"]
        ]
        state.next_instructions = [
            init_info["instruction_1"],
            init_info["instruction_2"],
            init_info["instruction_3"]
        ]
        return state
