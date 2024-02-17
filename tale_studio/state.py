import json
from typing import List, Any, Optional

from dataclasses import dataclass, asdict, field

import torch


@dataclass
class State:
    name: str = ""
    synopsis: str = ""
    outline: str = ""
    novel_type: str = ""
    language: str = ""
    description: str = ""
    paragraphs: List[str] = field(default_factory=lambda: list())
    l1_summaries: List[Any] = field(default_factory=lambda: list())
    l2_summaries: List[Any] = field(default_factory=lambda: list())
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

    def save(self, file_name):
        with open(file_name, "w") as w:
            json.dump(self.to_dict(), w, ensure_ascii=False)

    @classmethod
    def load(cls, file_name):
        with open(file_name, "r") as r:
            return cls.from_dict(json.load(r))
