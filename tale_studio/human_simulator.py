import json

from tale_studio.recurrentgpt import State
from tale_studio.utils import novel_json_completion, encode_prompt


class Human:
    def __init__(self, model_name: str, prompt_template: str, api_key: str = None):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.api_key = api_key

    def select_plan(self, state: State):
        prompt = encode_prompt(
            "human_select.jinja",
            previous_paragraph=state.paragraphs[-2],
            memory=state.short_memory,
            writer_new_paragraph=state.paragraphs[-1],
            previous_plans=state.next_instructions
        )
        print("HUMAN SELECT")
        print(prompt)
        print()
        output = self._complete(prompt)
        print("HUMAN SELECT RESPONSE")
        print(json.dumps(output, ensure_ascii=False, indent=4))
        print("==========")

        return output["selected_plan"]

    def step(self, state: State):
        state.instruction = self.select_plan(state)
        prompt = encode_prompt(
            "human_write.jinja",
            previous_paragraph=state.paragraphs[-2],
            memory=state.short_memory,
            writer_new_paragraph=state.paragraphs[-1],
            user_edited_plan=state.instruction
        )
        print("HUMAN STEP")
        print(prompt)
        print()
        output = self._complete(prompt)
        print("HUMAN STEP RESPONSE")
        print(json.dumps(output, ensure_ascii=False, indent=4))
        print("==========")

        extended_paragraph = output["extended_paragraph"]
        extended_paragraph = " ".join([p for p in extended_paragraph.split("\n") if p])
        extended_paragraph = extended_paragraph.strip()

        state.paragraphs = state.paragraphs[:-1] + [extended_paragraph]
        state.instruction = output["revised_plan"]
        return state

    def _complete(self, prompt):
        return novel_json_completion(
            prompt,
            model_name=self.model_name,
            prompt_template=self.prompt_template,
            api_key=self.api_key
        )
