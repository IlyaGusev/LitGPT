import re
import random

import gradio as gr
from sentence_transformers import SentenceTransformer

from recurrentgpt import RecurrentGPT, State
from human_simulator import Human
from utils import get_init

_CACHE = {}


INIT_PARAGRAPHS_TEMPLATE = """Title: {name}

Outline: {outline}

Paragraphs:

{first_paragraphs}"""

embedder = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
writer = RecurrentGPT(embedder=embedder)
#human = Human(embedder=embedder)



def init(novel_type, description):
    global _CACHE
    cookie = "1"
    init_info = get_init(novel_type, description)
    first_paragraphs = '\n\n'.join([init_info["paragraph_1"], init_info["paragraph_2"], init_info["paragraph_3"]])
    long_memory = [init_info["paragraph_1"], init_info['paragraph_2']]
    written_paragraphs = INIT_PARAGRAPHS_TEMPLATE.format(
        name=init_info["name"],
        outline=init_info["outline"],
        first_paragraphs=first_paragraphs
    )
    next_instructions = [init_info["instruction_1"], init_info["instruction_2"], init_info["instruction_3"]]
    state = State(
        last_paragraph=init_info["paragraph_3"],
        short_memory=init_info["summary"],
        long_memory=long_memory,
        written_paragraphs=written_paragraphs,
        next_instructions=next_instructions
    )

    _CACHE[cookie] = {"state": state}
    return (
        state.short_memory,
        state.format_long_memory(),
        state.written_paragraphs,
        state.next_instructions[0],
        state.next_instructions[1],
        state.next_instructions[2],
    )


def controled_step(short_memory, long_memory, selected_instruction, written_paragraphs):
    global _CACHE
    cookie = "1"
    cache = _CACHE[cookie]

    state = cache["state"]
    state.parse_long_memory(long_memory)
    state.instruction = selected_instruction
    state.short_memory = short_memory
    state.written_paragraphs = written_paragraphs
    writer.step(state)
    return (
        state.short_memory,
        state.format_long_memory(),
        state.written_paragraphs,
        state.next_instructions[0],
        state.next_instructions[1],
        state.next_instructions[2],
    )


def on_select(instruction1, instruction2, instruction3, evt: gr.SelectData):
    selected_plan = int(evt.value.replace("Instruction ", ""))
    selected_plan = [instruction1, instruction2, instruction3][selected_plan - 1]
    return selected_plan


with gr.Blocks(title="RecurrentGPT", css="footer {visibility: hidden}", theme="default") as demo:
    gr.Markdown(
        """
    # RecurrentGPT
    Interactive Generation of (Arbitrarily) Long Texts with Human-in-the-Loop
    """)
    with gr.Tab("Human-in-the-Loop"):
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    with gr.Row():
                        with gr.Column(scale=1, min_width=200):
                            novel_type = gr.Textbox(
                                label="Novel Type", placeholder="e.g. science fiction")
                        with gr.Column(scale=2, min_width=400):
                            description = gr.Textbox(label="Description")
                btn_init = gr.Button(
                    "Init Novel Generation", variant="primary")
                gr.Examples(["Science Fiction", "Romance", "Mystery", "Fantasy",
                            "Historical", "Horror", "Thriller", "Western", "Young Adult", ], inputs=[novel_type])
                written_paragraphs = gr.Textbox(
                    label="Written Paragraphs (editable)", max_lines=23, lines=23)
            with gr.Column():
                with gr.Box():
                    gr.Markdown("### Memory Module\n")
                    short_memory = gr.Textbox(
                        label="Short-Term Memory (editable)", max_lines=3, lines=3)
                    long_memory = gr.Textbox(
                        label="Long-Term Memory (editable)", max_lines=6, lines=6)
                with gr.Box():
                    gr.Markdown("### Instruction Module\n")
                    with gr.Row():
                        instruction1 = gr.Textbox(
                            label="Instruction 1", max_lines=3, lines=3, interactive=False)
                        instruction2 = gr.Textbox(
                            label="Instruction 2", max_lines=3, lines=3, interactive=False)
                        instruction3 = gr.Textbox(
                            label="Instruction 3", max_lines=3, lines=3, interactive=False)
                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            selected_plan = gr.Radio(["Instruction 1", "Instruction 2", "Instruction 3"], label="Instruction Selection",)
                                                    #  info="Select the instruction you want to revise and use for the next step generation.")
                        with gr.Column(scale=3, min_width=300):
                            selected_instruction = gr.Textbox(
                                label="Selected Instruction (editable)", max_lines=5, lines=5)

                btn_step = gr.Button("Next Step", variant="primary")

        btn_init.click(
            init,
            inputs=[novel_type, description],
            outputs=[short_memory, long_memory, written_paragraphs, instruction1, instruction2, instruction3]
        )
        btn_step.click(
            controled_step,
            inputs=[short_memory, long_memory, selected_instruction, written_paragraphs],
            outputs=[short_memory, long_memory, written_paragraphs, instruction1, instruction2, instruction3]
        )
        selected_plan.select(
            on_select,
            inputs=[instruction1, instruction2, instruction3], outputs=[selected_instruction]
        )

    demo.queue(concurrency_count=1)


if __name__ == "__main__":
    demo.launch(
        server_port=8005,
        share=True,
        server_name="0.0.0.0",
        show_api=False
    )
