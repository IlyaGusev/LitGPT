import json

from litgpt.recurrentgpt import State
from litgpt.utils import novel_json_completion, encode_prompt


class Human:
    def __init__(self, model_name):
        self.model_name = model_name

    def select_plan(self, state: State):
        prompt = encode_prompt(
            "human_select.jinja",
            previous_paragraph=state.prev_paragraph,
            memory=state.short_memory,
            writer_new_paragraph=state.last_paragraph,
            previous_plans=state.next_instructions
        )
        print("HUMAN SELECT")
        print(prompt)
        print()

        output = novel_json_completion(prompt, model_name=self.model_name)
        print("HUMAN SELECT RESPONSE")
        print(json.dumps(output, ensure_ascii=False, indent=4))
        print("==========")

        return output["selected_plan"]

    def step(self, state: State):
        state.instruction = self.select_plan(state)
        prompt = encode_prompt(
            "human_write.jinja",
            previous_paragraph=state.prev_paragraph,
            memory=state.short_memory,
            writer_new_paragraph=state.last_paragraph,
            user_edited_plan=state.instruction
        )
        print("HUMAN STEP")
        print(prompt)
        print()

        output = novel_json_completion(prompt, model_name=self.model_name)
        print("HUMAN STEP RESPONSE")
        print(json.dumps(output, ensure_ascii=False, indent=4))
        print("==========")

        extended_paragraph = output["extended_paragraph"]
        extended_paragraph = " ".join([p for p in extended_paragraph.split("\n") if p])
        extended_paragraph = extended_paragraph.strip()

        state.last_paragraph = extended_paragraph
        state.written_paragraphs = "\n\n".join(
            state.written_paragraphs.strip().split("\n\n")[:-1] + [extended_paragraph]
        )
        state.instruction = output["revised_plan"]
        return state
