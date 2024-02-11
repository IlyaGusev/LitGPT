import os
import fire
from nltk.tokenize import sent_tokenize

from tale_studio.openai_wrapper import openai_tokenize
from tale_studio.utils import novel_json_completion, encode_prompt
from tale_studio.model_settings import ModelSettings
from tale_studio.recurrentgpt import State


def tokenize(text: str, model_name: str):
    return openai_tokenize(model_name=model_name, text=text)


def extract_meta(paragraphs, model_settings):
    text = "\n\n".join(paragraphs)
    prompt = encode_prompt(
        os.path.join("existing_book", "extract_meta"),
        text=text
    )
    print("META PROMPT")
    print(prompt)
    print("========")
    output = novel_json_completion(prompt, model_settings=model_settings)
    print("META OUTPUT")
    print(output)
    print("========")

    state = State()
    state.name = output["name"]
    state.language = output["language"]
    return state


def summarize_paragraphs(state, prev_summary, paragraphs, model_settings):
    text = "\n\n".join(paragraphs)
    prompt = encode_prompt(
        os.path.join("existing_book", "summarize_paragraphs"),
        summary=prev_summary,
        text=text,
        language=state.language
    )
    print("PROMPT")
    print(prompt)
    print("========")
    output = novel_json_completion(prompt, model_settings=model_settings)
    print("OUTPUT")
    print(output)
    print("========")
    return output["summary"]


def summarize_book(
    input_file: str,
    output_file: str,
    language: str,
    model_name: str,
    min_paragraph_length: int = 400,
    max_paragraph_length: int = 1000,
    input_tokens_limit: int = 3000,
):
    assert input_file.endswith(".txt")

    with open(input_file) as r:
        text = r.read()

    paragraphs = [p for p in text.split("\n") if p.strip()]

    new_paragraphs = []
    for p in paragraphs:
        if len(p) < max_paragraph_length:
            new_paragraphs.append(p)
            continue

        num_parts = len(p) // max_paragraph_length + 1
        part_length = len(p) // num_parts
        current_paragraph = ""
        for sentence in sent_tokenize(p, language=language):
            if not current_paragraph:
                current_paragraph = sentence
                continue
            new_paragraph = " ".join((current_paragraph, sentence))
            if len(new_paragraph) > part_length:
                new_paragraphs.append(current_paragraph)
                current_paragraph = sentence
                continue
            current_paragraph = new_paragraph
        if current_paragraph:
            new_paragraphs.append(current_paragraph)
    paragraphs = new_paragraphs

    new_paragraphs = []
    current_paragraph = ""
    for p in paragraphs:
        if len(p) > min_paragraph_length:
            if current_paragraph:
                new_paragraphs.append(current_paragraph)
                current_paragraph = ""
            new_paragraphs.append(p)
            continue
        if not current_paragraph:
            current_paragraph = p
            continue
        current_paragraph = "\n".join((current_paragraph, p))
        if len(current_paragraph) > min_paragraph_length:
            new_paragraphs.append(current_paragraph)
            current_paragraph = ""
    if current_paragraph:
        new_paragraphs.append(current_paragraph)
    paragraphs = new_paragraphs

    state = None
    model_settings = ModelSettings(model_name=model_name)
    current_input = []
    current_input_tokens_count = 0
    summary = ""
    summaries = []
    for p in paragraphs:
        paragraph_tokens_count = len(tokenize(p, model_name=model_name))
        if current_input_tokens_count + paragraph_tokens_count < input_tokens_limit:
            current_input_tokens_count += paragraph_tokens_count
            current_input.append(p)
            continue

        if state is None:
            state = extract_meta(current_input, model_settings=model_settings)
        state.paragraphs += current_input

        summary = summarize_paragraphs(
            state=state,
            prev_summary=summary,
            paragraphs=current_input,
            model_settings=model_settings
        )
        summaries.extend(summary)
        current_input = [p]
        current_input_tokens_count = paragraph_tokens_count

    return current_summary

if __name__ == "__main__":
    fire.Fire(summarize_book)
