import random
import os
import gradio as gr

from litgpt.recurrentgpt import RecurrentGPT, State, gen_init_state
from litgpt.utils import encode_prompt, OPENAI_MODELS
from litgpt.human_simulator import Human
from litgpt.files import LOCAL_MODELS_LIST

MODEL_LIST = list(OPENAI_MODELS) + list(LOCAL_MODELS_LIST)
DEFAULT_MODEL_NAME = "gpt-3.5-turbo-16k"
EMBEDDER_LIST = [
    "embaas/sentence-transformers-multilingual-e5-base",
    "sentence-transformers/multi-qa-mpnet-base-cos-v1"
]
DEFAULT_EMBEDDER_NAME = "embaas/sentence-transformers-multilingual-e5-base"
DEFAULT_NOVEL_TYPE = "Science Fiction"
DEFAULT_DESCRIPTION = "Рассказ на русском в сеттинге коммунизма в высокотехнологичном будущем"


def init(state, novel_type, description, model_name):
    state = gen_init_state(
        novel_type=novel_type,
        description=description,
        model_name=model_name
    )
    return (
        state,
        state.name,
        state.global_summary,
        state.global_plan,
        state.short_memory,
        "\n\n".join(state.paragraphs),
        state.next_instructions[0],
        state.next_instructions[1],
        state.next_instructions[2],
    )


def step(
    state,
    global_plan,
    short_memory,
    paragraphs,
    selected_instruction,
    model_name,
    embedder_name,
    selection_mode
):
    writer = RecurrentGPT(embedder_name=embedder_name, model_name=model_name)

    assert state is not None
    state.instruction = selected_instruction
    state.short_memory = short_memory
    state.paragraphs = [p.strip() for p in paragraphs.split("\n\n") if p.strip()]

    if selection_mode == "gpt":
        human = Human(model_name=model_name)
        state = human.step(state)
    elif selection_mode == "random":
        state.instruction = random.choice(state.next_instructions)
    else:
        assert selected_instruction

    state = writer.step(state)

    return (
        state,
        state.global_plan,
        state.short_memory,
        "\n\n".join(state.paragraphs),
        state.next_instructions[0],
        state.next_instructions[1],
        state.next_instructions[2],
        ""
    )


def on_selected_plan_select(instruction1, instruction2, instruction3, evt: gr.SelectData):
    selected_plan = int(evt.value.replace("Instruction ", ""))
    selected_plan = [instruction1, instruction2, instruction3][selected_plan - 1]
    return selected_plan


def on_selection_mode_select(evt: gr.SelectData):
    value = evt.value
    is_manual = "manual" in value
    return (
        gr.Radio.update(interactive=is_manual),
        gr.Textbox.update(interactive=is_manual)
    )


with gr.Blocks(title="TaleStudio", css="footer {visibility: hidden}", theme="default") as demo:
    state = gr.State(None)
    gr.Markdown("# Tale Studio")
    with gr.Box():
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                model_name = gr.Dropdown(
                    MODEL_LIST,
                    value=DEFAULT_MODEL_NAME,
                    multiselect=False,
                    label="Model name",
                )
                embedder_name = gr.Dropdown(
                    EMBEDDER_LIST,
                    value=DEFAULT_EMBEDDER_NAME,
                    multiselect=False,
                    label="Embedder name",
                )
            with gr.Column(scale=1, min_width=200):
                novel_type = gr.Textbox(
                    label="Novel type",
                    value=DEFAULT_NOVEL_TYPE
                )
                gr.Examples([
                    "Science Fiction",
                    "Romance",
                    "Mystery",
                    "Fantasy",
                    "Historical",
                    "Horror",
                    "Thriller",
                    "Western",
                    "Young Adult"
                ], inputs=[novel_type])
            with gr.Column(scale=3, min_width=400):
                description = gr.Textbox(label="Description", value=DEFAULT_DESCRIPTION)
                gr.Examples([
                    "A novel about aliens",
                    "A love story of a man and AI",
                    "Dystopian society with a twist",
                    "Contemporary coming-of-age story",
                    "Magical realism in a small American town",
                    "Рассказ на русском в сеттинге коммунизма в высокотехнологичном будущем",
                    "История на русском об Англии 19 века и колониализме"
                ], inputs=[description])
    with gr.Row():
        btn_init = gr.Button(
            "Init Novel Generation",
            variant="primary"
        )

    with gr.Row():
        with gr.Column(scale=1):
            name = gr.Textbox(
                label="Name",
                max_lines=3,
                lines=3
            )
        with gr.Column(scale=4):
            global_summary = gr.Textbox(
                label="Global summary",
                max_lines=3,
                lines=3
            )

    with gr.Row():
        with gr.Column():
            paragraphs = gr.Textbox(
                label="Written Paragraphs (editable)",
                max_lines=25,
                lines=25
            )
        with gr.Column():
            global_plan = gr.Textbox(
                label="Global plan (editable)",
                max_lines=15,
                lines=16
            )
            short_memory = gr.Textbox(
                label="Short-Term Memory (editable)",
                max_lines=5,
                lines=6
            )

    with gr.Box():
        gr.Markdown("### Instruction Module\n")
        with gr.Row():
            instruction1 = gr.Textbox(
                label="Instruction 1", max_lines=7, lines=7, interactive=False
            )
            instruction2 = gr.Textbox(
                label="Instruction 2", max_lines=7, lines=7, interactive=False
            )
            instruction3 = gr.Textbox(
                label="Instruction 3", max_lines=7, lines=7, interactive=False
            )
        with gr.Row():
            selection_mode = gr.Radio(
                [("Select with GPT", "gpt"), ("Select randomly", "random"), ("Select manually", "manual")],
                label="Selection mode",
                value="random"
            )
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                selected_plan = gr.Radio(
                    ["Instruction 1", "Instruction 2", "Instruction 3"],
                    label="Instruction Selection",
                    interactive=False
                )
            with gr.Column(scale=3, min_width=300):
                selected_instruction = gr.Textbox(
                    label="Selected Instruction (editable)",
                    max_lines=5,
                    lines=5,
                    interactive=False
                )

    with gr.Row():
        btn_step = gr.Button("Next Step", variant="primary")

    btn_init.click(
        init,
        inputs=[state, novel_type, description, model_name],
        outputs=[
            state,
            name,
            global_summary,
            global_plan,
            short_memory,
            paragraphs,
            instruction1,
            instruction2,
            instruction3
        ]
    )
    btn_step.click(
        step,
        inputs=[
            state,
            global_plan,
            short_memory,
            paragraphs,
            selected_instruction,
            model_name,
            embedder_name,
            selection_mode
        ],
        outputs=[
            state,
            global_plan,
            short_memory,
            paragraphs,
            instruction1,
            instruction2,
            instruction3,
            selected_instruction
        ]
    )
    selected_plan.select(
        on_selected_plan_select,
        inputs=[instruction1, instruction2, instruction3],
        outputs=[selected_instruction]
    )
    selection_mode.select(
        on_selection_mode_select,
        inputs=[],
        outputs=[selected_plan, selected_instruction]
    )
    demo.queue(concurrency_count=1)


if __name__ == "__main__":
    demo.launch(
        server_port=8006,
        share=True,
        server_name="0.0.0.0",
        show_api=False
    )
