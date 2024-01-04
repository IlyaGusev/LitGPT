
from utils import novel_completion

class Human:

    def __init__(self, input, memory, embedder):
        self.input = input
        if memory:
            self.memory = memory
        else:
            self.memory = self.input['output_memory']
        self.embedder = embedder
        self.output = {}


    def prepare_input(self):
        previous_paragraph = self.input["input_paragraph"]
        writer_new_paragraph = self.input["output_paragraph"]
        memory = self.input["output_memory"]
        user_edited_plan = self.input["output_instruction"]

        input_text = encode_prompt(
            "prompts/human_write.jinja",
            previous_paragraph=previous_paragraph,
            memory=memory,
            writer_new_paragraph=writer_new_paragraph,
            user_edited_plan=user_edited_plan
        )
        return input_text

    def parse_plan(self,response):
        plan = get_content_between_a_b('Selected Plan:','Reason',response)
        return plan

    def select_plan(self):
        previous_paragraph = self.input["input_paragraph"]
        writer_new_paragraph = self.input["output_paragraph"]
        memory = self.input["output_memory"]
        previous_plans = self.input["output_instruction"]
        previous_plans = parse_instructions(previous_plans)
        encode_prompt(
            "prompts/human_select.jinja",
            previous_paragraph=previous_paragraph,
            memory=memory,
            writer_new_paragraph=writer_new_paragraph,
            previous_plans=previous_plans
        )
        print(prompt+'\n'+'\n')

        response = novel_completion(prompt)

        plan = self.parse_plan(response)
        while plan == None:
            response = novel_completion(prompt)
            plan= self.parse_plan(response)

        return plan
        
    def parse_output(self, text):
        try:
            if text.splitlines()[0].startswith('Extended Paragraph'):
                new_paragraph = get_content_between_a_b(
                    'Extended Paragraph:', 'Selected Plan', text)
            else:
                new_paragraph = text.splitlines()[0]

            lines = text.splitlines()
            if lines[-1] != '\n' and lines[-1].startswith('Revised Plan:'):
                revised_plan = lines[-1][len("Revised Plan:"):]
            elif lines[-1] != '\n':
                revised_plan = lines[-1]

            output = {
                "output_paragraph": new_paragraph,
                # "selected_plan": selected_plan,
                "output_instruction": revised_plan,
                # "memory":self.input["output_memory"]
            }

            return output
        except:
            return None

    def step(self, response_file=None):

        prompt = self.prepare_input()
        print(prompt+'\n'+'\n')

        response = novel_completion(prompt)
        self.output = self.parse_output(response)
        if response_file:
            with open(response_file, 'a', encoding='utf-8') as f:
                f.write(f"Human's output here:\n{response}\n\n")
