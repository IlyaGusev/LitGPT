import json
from typing import List, Optional
from dataclasses import dataclass, asdict, field

import torch

from tale_studio.embedders import EmbeddersStorage
from tale_studio.utils import (
    novel_json_completion,
    encode_prompt,
    cos_sim,
    novel_completion,
)


@dataclass
class State:
    name: str = ""
    synopsis: str = ""
    outline: str = ""
    novel_type: str = ""
    language: str = ""
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

    def get_relevant_long_memory(
        self, instruction, long_memory, memory_index, top_k: int = 2
    ):
        instruction_embedding = self.embedder.encode(
            self.query_prefix + instruction, convert_to_tensor=True
        )
        memory_scores = cos_sim(instruction_embedding, memory_index)[0]
        top_k = min(top_k, len(long_memory))
        top_k_idx = torch.topk(memory_scores, k=top_k)[1]
        top_k_memory = [long_memory[idx] for idx in top_k_idx]
        return "\n".join(
            [
                f"Related Paragraphs {i+1}: {memory}"
                for i, memory in enumerate(top_k_memory)
            ]
        )

    def step(self, state: State):
        assert state.instruction

        state.update_index(self.embedder, self.passage_prefix)
        formatted_long_memory = self.get_relevant_long_memory(
            state.instruction, state.long_memory, state.memory_index
        )

        output_paragraph = self._complete_text(
            "output",
            outline=state.outline,
            language=state.language,
            short_memory=state.short_memory,
            input_paragraph=state.paragraphs[-1],
            input_instruction=state.instruction,
            input_long_term_memory=formatted_long_memory,
        )
        output_paragraph = " ".join(
            [p.strip() for p in output_paragraph.split("\n") if p.strip()]
        )
        state.paragraphs.append(output_paragraph)
        state.update_index(self.embedder, self.passage_prefix)

        state.short_memory = self._complete_json(
            "summarize",
            language=state.language,
            short_memory=state.short_memory,
            input_paragraph=state.paragraphs[-2],
        )["updated_memory"]

        output = self._complete_json(
            "instruct",
            language=state.language,
            short_memory=state.short_memory,
            output_paragraph=state.paragraphs[-1],
            outline=state.outline,
            input_long_term_memory=formatted_long_memory,
        )
        state.next_instructions = [
            output["instruction_1"].strip(),
            output["instruction_2"].strip(),
            output["instruction_3"].strip(),
        ]

        return state

    def generate_name(self, state: State):
        return self._complete_json(
            "name",
            novel_type=state.novel_type,
            description=state.description,
            synopsis=state.synopsis,
            outline=state.outline,
            language=state.language,
        )["name"]

    def generate_meta(
        self,
        description: str,
        novel_type: str,
    ):
        while True:
            try:
                info = self._complete_json(
                    "meta", description=description, novel_type=novel_type
                )
                outline = info["outline"]

                assert isinstance(outline, list)
                assert outline
                keys = ("index", "chapter_name", "chapter_summary")
                assert all(key in outline[0] for key in keys)

                template = "Chapter {index}: {chapter_name}. {chapter_summary}"
                chapters = [template.format(**ch) for ch in outline]
                outline = "\n".join(chapters)

                break
            except AssertionError:
                continue

        return State(
            name=info["name"],
            synopsis=info["synopsis"],
            language=info["language"],
            outline=outline,
            novel_type=novel_type,
            description=description,
        )

    def generate_first_step(self, state: State):
        outline_start = state.outline.split("\n")[0]
        paragraphs = self._complete_text(
            "first_paragraphs",
            language=state.language,
            novel_type=state.novel_type,
            outline=outline_start,
            name=state.name,
            synopsis=state.synopsis,
        )
        paragraphs = paragraphs.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        state.paragraphs = paragraphs

        info = self._complete_json(
            "first_summary",
            novel_type=state.novel_type,
            outline=state.outline,
            language=state.language,
            name=state.name,
            synopsis=state.synopsis,
            paragraphs="\n\n".join(state.paragraphs),
        )
        state.short_memory = info["summary"]
        state.next_instructions = [
            info["instruction_1"],
            info["instruction_2"],
            info["instruction_3"],
        ]
        return state

    def _complete_json(self, prompt_name, **kwargs):
        prompt = encode_prompt(prompt_name, **kwargs)
        print(f"{prompt_name.upper()} PROMPT")
        print(prompt)
        print()
        result = novel_json_completion(prompt, model_settings=self.model_settings)
        print(f"{prompt_name.upper()} OUTPUT")
        print(json.dumps(result, ensure_ascii=False, indent=4))
        print("===========")
        return result

    def _complete_text(self, prompt_name, **kwargs):
        prompt = encode_prompt(prompt_name, **kwargs)
        print(f"{prompt_name.upper()} PROMPT")
        print(prompt)
        print()
        result = novel_completion(prompt, model_settings=self.model_settings)
        print(f"{prompt_name.upper()} OUTPUT")
        print(result)
        print("===========")
        return result
