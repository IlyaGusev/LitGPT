import json
import random
import os
import gradio as gr

import fire

from tale_studio.recurrentgpt import RecurrentGPT, State
from tale_studio.embedders import EMBEDDER_LIST
from tale_studio.utils import OPENAI_MODELS
from tale_studio.human_simulator import Human
from tale_studio.files import LOCAL_MODELS_LIST, SAVES_DIR_PATH
from tale_studio.prompt_templates import PROMPT_TEMPLATE_LIST, PROMPT_TEMPLATES, DEFAULT_PROMPT_TEMPLATE_NAME
from tale_studio.model_settings import ModelSettings


API_KEY = os.getenv("OPENAI_API_KEY", None)
MODEL_LIST = list(OPENAI_MODELS) + list(LOCAL_MODELS_LIST)
DEFAULT_NOVEL_TYPE = "Science Fiction"
DEFAULT_DESCRIPTION = "Рассказ на русском языке в сеттинге коммунизма в высокотехнологичном будущем"


def validate_inputs(model_state):
    if model_state.prompt_template == "openai" and model_state.model_name not in OPENAI_MODELS:
        raise gr.Error("Please set the correct prompt template!")
    if model_state.model_name in OPENAI_MODELS and not API_KEY and not model_state.api_key:
        raise gr.Error("Please set the API key!")


def generate_name(state, model_state):
    writer = RecurrentGPT(model_state)
    state.name = writer.generate_name(state)
    return (state, state.name)


def generate_meta(novel_type, description, model_state):
    validate_inputs(model_state)
    writer = RecurrentGPT(model_state)
    state = writer.generate_meta(novel_type=novel_type, description=description)
    return (state, state.name, state.language, state.synopsis, state.outline)


def generate_first_step(state, model_state):
    assert state is not None
    validate_inputs(model_state)
    writer = RecurrentGPT(model_state)
    state = writer.generate_first_step(state)
    return (
        state,
        state.short_memory,
        "\n\n".join(state.paragraphs),
        state.next_instructions[0],
        state.next_instructions[1],
        state.next_instructions[2],
    )


def step(state, model_state, selection_mode):
    assert state is not None
    validate_inputs(model_state)
    writer = RecurrentGPT(model_state)

    if selection_mode == "gpt":
        human = Human(model_state)
        state = human.step(state)
    elif selection_mode == "random":
        state.instruction = random.choice(state.next_instructions)
    else:
        assert instruction

    state = writer.step(state)

    return (
        state,
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
):
    if not file_name:
        raise gr.Error("File name should not be empty")
    if not name:
        raise gr.Error("Please set a name of the story")

    with open(os.path.join(root_dir, file_name), "w") as w:
        json.dump(state.to_dict(), w, ensure_ascii=False, indent=4)


def load(file_name):
    with open(file_name) as r:
        state = State.from_dict(json.load(r))
    return (
        state,
        state.name,
        state.synopsis,
        state.outline,
        state.short_memory,
        "\n\n".join(state.paragraphs),
        state.next_instructions[0],
        state.next_instructions[1],
        state.next_instructions[2],
    )


def load_from_saves(file_name):
    full_path = os.path.join(SAVES_DIR_PATH, file_name)
    return load(full_path)


def get_saves_list():
    files = os.listdir(SAVES_DIR_PATH)
    files = [f for f in files if not f.startswith(".")]
    first_file = files[0] if files else None
    return gr.update(choices=files, value=first_file, interactive=True)


def on_selected_plan_select(instruction1, instruction2, instruction3, evt: gr.SelectData):
    selected_plan = int(evt.value.replace("Instruction ", ""))
    selected_plan = [instruction1, instruction2, instruction3][selected_plan - 1]
    return selected_plan


def on_selection_mode_select(evt: gr.SelectData):
    value = evt.value
    is_manual = "manual" in value
    return gr.Row.update(visible=is_manual)


def on_model_name_select(model_state, evt: gr.SelectData):
    value = evt.value
    is_local = "gguf" in value
    model_state.model_name = value
    return gr.update(visible=not is_local), model_state


def on_prompt_template_name_select(model_state, prompt_template, evt: gr.SelectData):
    value = evt.value
    is_custom = "custom" in value
    prompt_template = PROMPT_TEMPLATES[value]
    is_openai = "openai" in value
    model_state.prompt_template = prompt_template
    return model_state, gr.update(value=prompt_template, interactive=is_custom, visible=not is_openai)


with gr.Blocks(title="TaleStudio", css="footer {visibility: hidden}") as demo:
    state = gr.State(State())
    model_state = gr.State(ModelSettings())
    gr.Markdown("# Tale Studio")
    with gr.Tab("Main"):
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                with gr.Group():
                    novel_type = gr.Textbox(
                        label="Novel genre and style",
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
                        "Рассказ на русском языке в сеттинге коммунизма в высокотехнологичном будущем",
                        "История на русском языке об Англии 19 века и колониализме"
                    ], inputs=[description])
        with gr.Row():
            btn_init = gr.Button(
                "Start New Novel",
                variant="primary"
            )

        with gr.Row():
            with gr.Column(scale=5):
                name = gr.Textbox(
                    label="Name (editable)",
                    max_lines=1,
                    lines=1
                )
                btn_gen_name = gr.Button(
                    "Regenerate",
                    variant="primary"
                )
                language = gr.Textbox(
                    label="Language (editable)",
                    max_lines=1,
                    lines=1
                )
                synopsis = gr.Textbox(
                    label="Synopsis (editable)",
                    max_lines=8,
                    lines=8
                )
            with gr.Column(scale=9):
                outline = gr.Textbox(
                    label="Outline (editable)",
                    lines=17,
                    max_lines=17
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
            instruction = gr.Textbox(
                label="Selected Instruction (editable)",
                max_lines=5,
                lines=5,
            )

        with gr.Row():
            btn_step = gr.Button("Next Step", variant="primary")
        with gr.Row() as save_load_buttons:
            btn_save = gr.Button("Save", variant="primary")
            btn_load = gr.Button("Load", variant="primary")
            btn_upload = gr.UploadButton("Upload", variant="primary")

        with gr.Group(visible=False) as file_saver:
            save_filename = gr.Textbox(lines=1, label="File name")
            save_root = gr.Textbox(
                lines=1,
                label="File folder",
                info="For reference. Unchangeable.",
                interactive=False,
                value=SAVES_DIR_PATH
            )
            with gr.Row():
                btn_confirm_save = gr.Button("Confirm", variant="primary")
                btn_close_save = gr.Button("Close", variant="secondary")

        with gr.Group(visible=False) as file_loader:
            load_filename = gr.Dropdown(label="File name", choices=[], value=None)
            with gr.Row():
                btn_confirm_load = gr.Button("Confirm", variant="primary")
                btn_close_load = gr.Button("Close", variant="secondary")

    with gr.Tab("Model"):

        with gr.Group():
            with gr.Row():
                with gr.Column(scale=1, min_width=200):
                    model_name = gr.Dropdown(
                        MODEL_LIST,
                        value=model_state.value.model_name,
                        multiselect=False,
                        label="Model name",
                    )
                with gr.Column(scale=1, min_width=200):
                    embedder_name = gr.Dropdown(
                        EMBEDDER_LIST,
                        value=model_state.value.embedder_name,
                        multiselect=False,
                        label="Embedder name",
                    )
        with gr.Group():
            with gr.Row():
                api_key = gr.Textbox(
                    label="API key",
                    value="",
                )
            with gr.Row():
                prompt_template_name = gr.Dropdown(
                    PROMPT_TEMPLATE_LIST,
                    value=DEFAULT_PROMPT_TEMPLATE_NAME,
                    multiselect=False,
                    label="Prompt template name",
                    interactive=True
                )
            with gr.Row():
                prompt_template = gr.Textbox(
                    label="Prompt template text",
                    value=PROMPT_TEMPLATES[DEFAULT_PROMPT_TEMPLATE_NAME],
                    interactive=False,
                    visible=False
                )

        with gr.Group():
            with gr.Row():
                with gr.Column(scale=1, min_width=200):
                    temperature = gr.Slider(
                        minimum=0.01,
                        maximum=1.50,
                        value=model_state.value.generation_params.temperature,
                        step=0.01,
                        interactive=True,
                        label="Temperature"
                    )
                    repetition_penalty = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=model_state.value.generation_params.repetition_penalty,
                        step=0.05,
                        interactive=True,
                        label="Repetition penalty"
                    )
                with gr.Column(scale=1, min_width=200):
                    top_p = gr.Slider(
                        minimum=0.01,
                        maximum=1.0,
                        value=model_state.value.generation_params.top_p,
                        step=0.05,
                        interactive=True,
                        label="Top-p",
                    )
                    top_k = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=model_state.value.generation_params.top_k,
                        step=5,
                        interactive=True,
                        label="Top-k",
                    )

    # Sync inputs
    def set_field(state, field, value):
        setattr(state, field, value)
        return state

    state_fields = {
        "name": name,
        "language": language,
        "description": description,
        "novel_type": novel_type,
        "synopsis": synopsis,
        "outline": outline,
        "instruction": instruction,
        "short_memory": short_memory
    }
    for key, field in state_fields.items():
        field.change((lambda s, f, k=key: set_field(s, k, f)), [state, field], state)

    def set_paragraphs(state, paragraphs):
        state.paragraphs = [p.strip() for p in paragraphs.split("\n\n") if p.strip()]
        return state

    paragraphs.change(set_paragraphs, [state, paragraphs], state)

    model_state_fields = {
        "model_name": model_name,
        "prompt_template": prompt_template,
        "embedder_name": embedder_name,
        "api_key": api_key,
    }
    for key, field in model_state_fields.items():
        field.change((lambda s, f, k=key: set_field(s, k, f)), [model_state, field], model_state)

    def set_param(state, field, value):
        setattr(state.generation_params, field, value)
        return state

    generation_params_fields = {
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "top_k": top_k
    }
    for key, field in generation_params_fields.items():
        field.change((lambda s, f, k=key: set_param(s, k, f)), [model_state, field], model_state)

    # Main events
    btn_init.click(
        generate_meta,
        inputs=[novel_type, description, model_state],
        outputs=[state, name, language, synopsis, outline]
    ).success(
        generate_first_step,
        inputs=[state, model_state],
        outputs=[state, short_memory, paragraphs, instruction1, instruction2, instruction3]
    )

    btn_step.click(
        step,
        inputs=[state, model_state, selection_mode],
        outputs=[
            state,
            short_memory,
            paragraphs,
            instruction1,
            instruction2,
            instruction3,
            instruction
        ]
    )
    btn_gen_name.click(
        generate_name,
        inputs=[state, model_state],
        outputs=[state, name]
    )

    # Save/Load
    btn_save.click(
        lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[file_saver, save_load_buttons]
    )
    btn_confirm_save.click(
        save,
        inputs=[save_filename, save_root, state]
    ).success(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[file_saver, save_load_buttons]
    )
    btn_close_save.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[file_saver, save_load_buttons]
    )
    btn_load.click(
        get_saves_list,
        outputs=[load_filename]
    ).then(
        lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[file_loader, save_load_buttons]
    )
    btn_close_load.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[file_loader, save_load_buttons]
    )
    btn_confirm_load.click(
        load_from_saves,
        inputs=[load_filename],
        outputs=[
            state,
            name,
            synopsis,
            outline,
            short_memory,
            paragraphs,
            instruction1,
            instruction2,
            instruction3,
        ]
    ).success(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[file_loader, save_load_buttons]
    )

    btn_upload.upload(
        load,
        inputs=[btn_upload],
        outputs=[
            state,
            name,
            synopsis,
            outline,
            short_memory,
            paragraphs,
            instruction1,
            instruction2,
            instruction3,
        ]
    )

    # Other events
    selected_plan.select(
        on_selected_plan_select,
        inputs=[instruction1, instruction2, instruction3],
        outputs=[instruction]
    )
    selection_mode.select(
        on_selection_mode_select,
        inputs=[],
        outputs=[instruction_selection]
    )
    model_name.select(
        on_model_name_select,
        inputs=[model_state],
        outputs=[api_key, model_state]
    )
    prompt_template_name.select(
        on_prompt_template_name_select,
        inputs=[model_state, prompt_template],
        outputs=[model_state, prompt_template]
    )
    demo.queue()


def launch(
    server_port: int = 8080,
    server_name: str = "0.0.0.0",
    share: bool = False
):
    demo.launch(
        server_port=server_port,
        share=share,
        server_name=server_name,
        show_api=False,
        show_error=True,
        favicon_path="static/favicon.ico"
    )


if __name__ == "__main__":
    fire.Fire(launch)
