import gradio as gr
from sentence_transformers import SentenceTransformer

from recurrentgpt import RecurrentGPT, State, get_init
from utils import encode_prompt
from human_simulator import Human


embedder = SentenceTransformer('multi-qa-mpnet-base-cos-v1')


def init(state, novel_type, description, model_name):
    init_info = get_init(novel_type, description, model_name=model_name)
    first_paragraphs = '\n\n'.join([
        init_info["paragraph_1"],
        init_info["paragraph_2"],
        init_info["paragraph_3"]
    ])
    written_paragraphs = encode_prompt(
        "prompts/paragraphs.jinja",
        name=init_info["name"],
        outline=init_info["outline"],
        first_paragraphs=first_paragraphs
    )
    state = State(
        prev_paragraph=init_info["paragraph_2"],
        last_paragraph=init_info["paragraph_3"],
        short_memory=init_info["summary"],
        long_memory=[
            init_info["paragraph_1"],
            init_info['paragraph_2']
        ],
        written_paragraphs=written_paragraphs,
        next_instructions=[
            init_info["instruction_1"],
            init_info["instruction_2"],
            init_info["instruction_3"]
        ]
    )

    return (
        state,
        state.short_memory,
        state.format_long_memory(),
        state.written_paragraphs,
        state.next_instructions[0],
        state.next_instructions[1],
        state.next_instructions[2],
    )


def step(state, short_memory, long_memory, selected_instruction, written_paragraphs, model_name, emulate_human):
    writer = RecurrentGPT(embedder=embedder, model_name=model_name)

    assert state is not None
    state.parse_long_memory(long_memory)
    state.instruction = selected_instruction
    state.short_memory = short_memory
    state.written_paragraphs = written_paragraphs

    if emulate_human:
        human = Human(model_name=model_name)
        state = human.step(state)
    else:
        assert selected_instruction

    state = writer.step(state)

    return (
        state,
        state.short_memory,
        state.format_long_memory(),
        state.written_paragraphs,
        state.next_instructions[0],
        state.next_instructions[1],
        state.next_instructions[2],
        ""
    )


def on_select(instruction1, instruction2, instruction3, evt: gr.SelectData):
    selected_plan = int(evt.value.replace("Instruction ", ""))
    selected_plan = [instruction1, instruction2, instruction3][selected_plan - 1]
    return selected_plan


def update_state(emulate_human_value):
    return (
        gr.Radio.update(interactive=not emulate_human_value),
        gr.Textbox.update(interactive=not emulate_human_value)
    )


with gr.Blocks(title="RecurrentGPT", css="footer {visibility: hidden}", theme="default") as demo:
    state = gr.State(None)
    gr.Markdown(
        """
    # RecurrentGPT
    Interactive Generation of (Arbitrarily) Long Texts with Human-in-the-Loop
    """)
    with gr.Row():
        with gr.Column():
            with gr.Box():
                with gr.Row():
                    with gr.Column(scale=1, min_width=200):
                        model_name = gr.Dropdown(
                            ["gpt-3.5-turbo-16k", "gpt-4-1106-preview"],
                            value="gpt-3.5-turbo-16k",
                            multiselect=False,
                            label="Model name",
                        )
                    with gr.Column(scale=1, min_width=200):
                        novel_type = gr.Textbox(
                            label="Novel type", placeholder="e.g. science fiction"
                        )
                    with gr.Column(scale=2, min_width=400):
                        description = gr.Textbox(label="Description"
                    )
            btn_init = gr.Button(
                "Init Novel Generation",
                variant="primary"
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
            written_paragraphs = gr.Textbox(
                label="Written Paragraphs (editable)",
                max_lines=23,
                lines=23
            )
        with gr.Column():
            with gr.Box():
                gr.Markdown("### Memory Module\n")
                short_memory = gr.Textbox(
                    label="Short-Term Memory (editable)",
                    max_lines=3,
                    lines=3
                )
                long_memory = gr.Textbox(
                    label="Long-Term Memory (editable)",
                    max_lines=6,
                    lines=6
                )
            with gr.Box():
                gr.Markdown("### Instruction Module\n")
                with gr.Row():
                    instruction1 = gr.Textbox(
                        label="Instruction 1", max_lines=3, lines=3, interactive=False
                    )
                    instruction2 = gr.Textbox(
                        label="Instruction 2", max_lines=3, lines=3, interactive=False
                    )
                    instruction3 = gr.Textbox(
                        label="Instruction 3", max_lines=3, lines=3, interactive=False
                    )
                with gr.Row():
                    emulate_human = gr.Checkbox(
                        label="Automatic",
                        info="Select automatically"
                    )
                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        selected_plan = gr.Radio(
                            ["Instruction 1", "Instruction 2", "Instruction 3"],
                            label="Instruction Selection",
                        )
                    with gr.Column(scale=3, min_width=300):
                        selected_instruction = gr.Textbox(
                            label="Selected Instruction (editable)",
                            max_lines=5,
                            lines=5,
                        )
                emulate_human.change(update_state, inputs=[emulate_human], outputs=[selected_plan, selected_instruction])

            btn_step = gr.Button("Next Step", variant="primary")

    btn_init.click(
        init,
        inputs=[state, novel_type, description, model_name],
        outputs=[
            state,
            short_memory,
            long_memory,
            written_paragraphs,
            instruction1,
            instruction2,
            instruction3
        ]
    )
    btn_step.click(
        step,
        inputs=[
            state,
            short_memory,
            long_memory,
            selected_instruction,
            written_paragraphs,
            model_name,
            emulate_human
        ],
        outputs=[
            state,
            short_memory,
            long_memory,
            written_paragraphs,
            instruction1,
            instruction2,
            instruction3,
            selected_instruction
        ]
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
