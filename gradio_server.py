import json
import random
import os
import gradio as gr
from tempfile import NamedTemporaryFile

import fire

from litgpt.recurrentgpt import RecurrentGPT, State
from litgpt.utils import encode_prompt, OPENAI_MODELS
from litgpt.human_simulator import Human
from litgpt.files import LOCAL_MODELS_LIST
from litgpt.prompt_templates import PROMPT_TEMPLATE_LIST, DEFAULT_PROMPT_TEMPLATE_NAME

MODEL_LIST = list(OPENAI_MODELS) + list(LOCAL_MODELS_LIST)
DEFAULT_MODEL_NAME = "gpt-3.5-turbo-16k"
EMBEDDER_LIST = [
    "embaas/sentence-transformers-multilingual-e5-base",
    "sentence-transformers/multi-qa-mpnet-base-cos-v1"
]
DEFAULT_EMBEDDER_NAME = "embaas/sentence-transformers-multilingual-e5-base"
DEFAULT_NOVEL_TYPE = "Science Fiction"
DEFAULT_DESCRIPTION = "Рассказ на русском в сеттинге коммунизма в высокотехнологичном будущем"


def generate_plan(novel_type, description, model_name, prompt_template, embedder_name):
    writer = RecurrentGPT(
        embedder_name=embedder_name,
        model_name=model_name,
        prompt_template=prompt_template
    )
    state = writer.generate_plan(novel_type=novel_type, description=description)
    return (state, state.name, state.synopsis, state.plan)



def generate_first_paragraphs(state, name, synopsis, plan, model_name, prompt_template, embedder_name):
    if prompt_template == "openai" and model_name not in OPENAI_MODELS:
        raise gr.Error("Please set correct prompt_template")

    writer = RecurrentGPT(
        embedder_name=embedder_name,
        model_name=model_name,
        prompt_template=prompt_template
    )

    state.name = name
    state.synopsis = synopsis
    state.plan = plan
    state = writer.generate_first_paragraphs(state)
    return (
        state,
        state.short_memory,
        "\n\n".join(state.paragraphs),
        state.next_instructions[0],
        state.next_instructions[1],
        state.next_instructions[2],
    )


def step(
    state,
    plan,
    short_memory,
    paragraphs,
    selected_instruction,
    model_name,
    prompt_template,
    embedder_name,
    selection_mode
):
    writer = RecurrentGPT(
        embedder_name=embedder_name,
        model_name=model_name,
        prompt_template=prompt_template
    )

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
        state.plan,
        state.short_memory,
        "\n\n".join(state.paragraphs),
        state.next_instructions[0],
        state.next_instructions[1],
        state.next_instructions[2],
        ""
    )


def save(
    file_name,
    root_dir,
    state,
    name,
    synopsis,
    plan,
    short_memory,
    paragraphs,
):
    state.name = name
    state.synopsis = synopsis
    state.plan = plan
    state.short_memory = short_memory
    state.paragraphs = [p.strip() for p in paragraphs.split("\n\n") if p.strip()]
    f = NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    with open(os.path.join(root_dir, file_name), "w") as w:
        json.dump(state.to_dict(), w, ensure_ascii=False, indent=4)


def on_selected_plan_select(instruction1, instruction2, instruction3, evt: gr.SelectData):
    selected_plan = int(evt.value.replace("Instruction ", ""))
    selected_plan = [instruction1, instruction2, instruction3][selected_plan - 1]
    return selected_plan


def on_selection_mode_select(evt: gr.SelectData):
    value = evt.value
    is_manual = "manual" in value
    return gr.Row.update(visible=is_manual)


with gr.Blocks(title="TaleStudio", css="footer {visibility: hidden}") as demo:
    state = gr.State(None)
    gr.Markdown("# Tale Studio")
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            with gr.Group():
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
                prompt_template = gr.Dropdown(
                    PROMPT_TEMPLATE_LIST,
                    value=DEFAULT_PROMPT_TEMPLATE_NAME,
                    multiselect=False,
                    label="Prompt template",
                )

        with gr.Column(scale=1, min_width=200):
            with gr.Group():
                novel_type = gr.Textbox(
                    label="Novel type",
                    value=DEFAULT_NOVEL_TYPE,
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
            with gr.Group():
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
                label="Name (editable)",
                max_lines=1,
                lines=1
            )
            synopsis = gr.Textbox(
                label="Synopsis (editable)",
                max_lines=8,
                lines=8
            )
        with gr.Column(scale=3):
            plan = gr.Textbox(
                label="Global plan (editable)",
                lines=13,
                max_lines=13
            )

    with gr.Group():
        with gr.Row():
            paragraphs = gr.Textbox(
                label="Written Paragraphs (editable)",
                max_lines=20,
                lines=20
            )
        with gr.Row():
            short_memory = gr.Textbox(
                label="Short-Term Memory (editable)",
                max_lines=5,
                lines=5
            )

    with gr.Group():
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

    with gr.Group(visible=False) as instruction_selection:
        selected_plan = gr.Radio(
            ["Instruction 1", "Instruction 2", "Instruction 3"],
            label="Instruction Selection",
        )
        selected_instruction = gr.Textbox(
            label="Selected Instruction (editable)",
            max_lines=5,
            lines=5,
        )

    with gr.Row():
        btn_step = gr.Button("Next Step", variant="primary")
    with gr.Row():
        btn_save = gr.Button("Save", variant="primary")

    with gr.Group(visible=False) as file_saver:
        save_filename = gr.Textbox(lines=1, label='File name')
        save_root = gr.Textbox(lines=1, label='File folder', info='For reference. Unchangeable.', interactive=False, value="saves/")
        btn_confirm_save = gr.Button("Confirm", variant="primary")

    btn_init.click(
        generate_plan,
        inputs=[novel_type, description, model_name, prompt_template, embedder_name],
        outputs=[state, name, synopsis, plan]
    ).then(
        generate_first_paragraphs,
        inputs=[state, name, synopsis, plan, model_name, prompt_template, embedder_name],
        outputs=[state, short_memory, paragraphs, instruction1, instruction2, instruction3]
    )

    btn_step.click(
        step,
        inputs=[
            state,
            plan,
            short_memory,
            paragraphs,
            selected_instruction,
            model_name,
            prompt_template,
            embedder_name,
            selection_mode
        ],
        outputs=[
            state,
            plan,
            short_memory,
            paragraphs,
            instruction1,
            instruction2,
            instruction3,
            selected_instruction
        ]
    )
    btn_save.click(
        lambda: gr.update(visible=True),
        outputs=[file_saver]
    )
    btn_confirm_save.click(
        save,
        inputs=[
            save_filename,
            save_root,
            state,
            name,
            synopsis,
            plan,
            short_memory,
            paragraphs,
        ]
    ).then(
        lambda: gr.update(visible=False),
        outputs=[file_saver]
    )
    selected_plan.select(
        on_selected_plan_select,
        inputs=[instruction1, instruction2, instruction3],
        outputs=[selected_instruction]
    )
    selection_mode.select(
        on_selection_mode_select,
        inputs=[],
        outputs=[instruction_selection]
    )
    demo.queue()


def launch(
    server_port: int = 8080,
    server_name: str = "0.0.0.0",
):
    demo.launch(
        server_port=server_port,
        share=True,
        server_name=server_name,
        show_api=False,
        show_error=True
    )


if __name__ == "__main__":
    fire.Fire(launch)
