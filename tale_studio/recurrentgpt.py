import json
from typing import List, Optional
from dataclasses import dataclass, asdict, field

import torch

from tale_studio.embedders import EmbeddersStorage
from tale_studio.utils import novel_json_completion, encode_prompt, cos_sim, novel_completion


@dataclass
class State:
    name: str = ""
    synopsis: str = ""
    plan: str = ""
    novel_type: str = ""
    language: str = "English"
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

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class RecurrentGPT:
    def __init__(self, model_settings):
        self.model_settings = model_settings
        self.embedder = EmbeddersStorage.get_embedder(model_settings.embedder_name)
        self.query_prefix = "query: "
        self.passage_prefix = "passage: "

    def get_relevant_long_memory(self, instruction, long_memory, memory_index, top_k: int = 2):
        instruction_embedding = self.embedder.encode(self.query_prefix + instruction, convert_to_tensor=True)
        memory_scores = cos_sim(instruction_embedding, memory_index)[0]
        top_k = min(top_k, len(long_memory))
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

        output_paragraph = self.output(
            plan=state.plan,
            language=state.language,
            short_memory=state.short_memory,
            input_paragraph=state.paragraphs[-1],
            input_instruction=state.instruction,
            input_long_term_memory=formatted_long_memory,
        )
        state.paragraphs.append(output_paragraph)
        state.update_index(self.embedder, self.passage_prefix)

        state.short_memory = self.summarize(
            language=state.language,
            short_memory=state.short_memory,
            input_paragraph=state.paragraphs[-2],
        )

        state.next_instructions = self.instruct(
            language=state.language,
            short_memory=state.short_memory,
            output_paragraph=state.paragraphs[-1],
            plan=state.plan,
            input_long_term_memory=formatted_long_memory,
        )

        return state

    def output(self, **kwargs):
        prompt = encode_prompt("output.jinja", **kwargs)
        print("OUTPUT PROMPT")
        print(prompt)
        print()
        output_paragraph = self._complete_text(prompt)
        output_paragraph = " ".join([p.strip() for p in output_paragraph.split("\n") if p.strip()])
        print("OUTPUT")
        print(output_paragraph)
        print("===========")
        return output_paragraph

    def summarize(self, **kwargs):
        prompt = encode_prompt("summarize.jinja", **kwargs)
        print("SUMMARIZE PROMPT")
        print(prompt)
        print()
        output = self._complete_json(prompt)
        print("SUMMARIZE OUTPUT")
        print(json.dumps(output, ensure_ascii=False, indent=4))
        print("===========")
        return output["updated_memory"]

    def instruct(self, **kwargs):
        prompt = encode_prompt("instruct.jinja", **kwargs)
        print("INSTRUCT PROMPT")
        print(prompt)
        print()
        output = self._complete_json(prompt)
        print("INSTRUCT OUTPUT")
        print(json.dumps(output, ensure_ascii=False, indent=4))
        print("===========")
        return [
            output["instruction_1"].strip(),
            output["instruction_2"].strip(),
            output["instruction_3"].strip(),
        ]

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
        plan_info = self._complete_json(plan_prompt)
        print("PLAN OUTPUT")
        print(json.dumps(plan_info, ensure_ascii=False, indent=4))
        print("===========")

        chapter_summaries = plan_info["chapter_summaries"]
        if isinstance(chapter_summaries, dict):
            chapter_summaries = [" ".join((k, v)) for k, v in chapter_summaries.items()]
        return State(
            name=plan_info["name"],
            synopsis=plan_info["synopsis"],
            plan="\n".join(chapter_summaries),
            novel_type=novel_type,
            description=description,
            language=plan_info["language"]
        )

    def generate_first_paragraphs(
        self,
        state: State
    ):
        plan_start = state.plan.split("\n")[0]
        paragraphs = self.begin(
            language=state.language,
            novel_type=state.novel_type,
            plan=plan_start,
            name=state.name,
            synopsis=state.synopsis,
        )
        state.paragraphs = paragraphs

        info = self.process(
            novel_type=state.novel_type,
            plan=state.plan,
            language=state.language,
            name=state.name,
            synopsis=state.synopsis,
            paragraphs="\n\n".join(state.paragraphs)
        )
        state.short_memory = info["summary"]
        state.next_instructions = [
            info["instruction_1"],
            info["instruction_2"],
            info["instruction_3"]
        ]
        return state

    def begin(self, **kwargs):
        begin_prompt = encode_prompt("begin.jinja", **kwargs)
        print("BEGIN PROMPT")
        print(begin_prompt)
        print()
        paragraphs = self._complete_text(begin_prompt).split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        print("BEGIN OUTPUT")
        print("\n\n".join(paragraphs))
        print("===========")
        return paragraphs

    def process(self, **kwargs):
        process_prompt = encode_prompt("process.jinja", **kwargs)
        print("PROCESS PROMPT")
        print(process_prompt)
        print()
        info = self._complete_json(process_prompt)
        print("PROCESS OUTPUT")
        print(json.dumps(info, ensure_ascii=False, indent=4))
        print("===========")
        return info

    def _complete_json(self, prompt):
        return novel_json_completion(
            prompt,
            model_settings=self.model_settings
        )

    def _complete_text(self, prompt):
        return novel_completion(
            prompt,
            model_settings=self.model_settings
        )
