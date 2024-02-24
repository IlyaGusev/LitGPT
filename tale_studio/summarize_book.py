import os
import copy
import fire
import json
from typing import List, Any

from nltk.tokenize import sent_tokenize

from tale_studio.utils import novel_json_completion, encode_prompt, tokenize
from tale_studio.model_settings import ModelSettings, GenerationParams
from tale_studio.state import State


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
    return (output["name"], output["language"])


def summarize(
    paragraphs: List[str],
    language: str,
    prev_summary: str = "",
    prev_chapter_header: str = "",
    model_settings: ModelSettings = ModelSettings(),
    prompt: str = "l1_summarize",
    num_sentences: int = 10
):
    text = "\n\n".join(paragraphs)
    prompt = encode_prompt(
        os.path.join("existing_book", prompt),
        prev_summary=prev_summary,
        prev_chapter_header=prev_chapter_header,
        text=text,
        language=language,
        num_sentences=num_sentences
    )
    print("PROMPT")
    print(prompt)
    print("========")
    output = novel_json_completion(prompt, model_settings=model_settings)
    print("OUTPUT")
    print(output)
    print("========")
    return output["summary"]


def split_paragrahps(paragraphs, max_paragraph_length, language):
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
    return new_paragraphs


def merge_paragraphs(paragraphs, min_paragraph_length):
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
    return new_paragraphs


def gen_windows(
    paragraphs: List[str],
    model_settings: ModelSettings,
    start_index: int = -1,
    input_tokens_limit: int = 2000
):
    window = []
    window_tokens_count = 0
    for pnum, p in enumerate(paragraphs):
        if pnum <= start_index:
            continue

        paragraph_tokens_count = len(tokenize(p, model_settings=model_settings))
        if window_tokens_count + paragraph_tokens_count < input_tokens_limit:
            window_tokens_count += paragraph_tokens_count
            window.append((pnum, p))
            continue

        yield window[:]

        window = [(pnum, p)]
        window_tokens_count = paragraph_tokens_count

    if window:
        yield window


def summarize_paragraphs_by_windows(
    paragraphs: List[str],
    model_settings: ModelSettings,
    cached_summaries: List[Any] = tuple(),
    prev_summary: str = "",
    input_tokens_limit: int = 2000,
    language: str = "English",
    prompt: str = "l1_summarize",
    num_sentences: int = 10
):
    prev_chapter_header = ""

    start_index = -1
    if cached_summaries and "paragraph_number" in cached_summaries[0]:
        start_index = max([s["paragraph_number"] for s in cached_summaries])

    for window in gen_windows(
        paragraphs,
        start_index=start_index,
        input_tokens_limit=input_tokens_limit,
        model_settings=model_settings
    ):
        texts = [p for _, p in window]
        if not "\n".join(texts).strip():
            continue
        summary = summarize(
            language=language,
            prev_summary=prev_summary,
            prev_chapter_header=prev_chapter_header,
            paragraphs=texts,
            model_settings=model_settings,
            prompt=prompt,
            num_sentences=num_sentences
        )
        if isinstance(summary, str):
            prev_summary = summary
            yield summary
            continue

        summary_points = [s for s in summary if "summary_point" in s]
        prev_summary = json.dumps({"summary": summary_points}, ensure_ascii=False)

        headers = [s for s in summary if "chapter_header" in s]
        if headers:
            prev_chapter_header = headers[-1]["chapter_header"]

        pnum = max([pnum for pnum, _ in window])
        for s in summary:
            s["paragraph_number"] = pnum
            yield s


def postprocess_l1(summaries):
    fixed_summaries = []
    summaries_set = set()
    for s in summaries:
        fixed_s = copy.deepcopy(s)
        fixed_s.pop("paragraph_number")
        if str(fixed_s) in summaries_set:
            continue
        summaries_set.add(str(fixed_s))
        fixed_summaries.append(s)

    summaries = []
    current_chapter_points_count = 0
    for s in fixed_summaries:
        if "chapter_header" not in s:
            current_chapter_points_count += 1
            summaries.append(s)
        elif current_chapter_points_count > 4:
            current_chapter_points_count = 0
            summaries.append(s)
    return summaries


def summarize_book(
    input_file: str,
    output_file: str,
    language: str,
    model_name: str,
    min_paragraph_length: int = 400,
    max_paragraph_length: int = 1000,
    input_tokens_limit: int = 2000,
):
    assert input_file.endswith(".txt")

    with open(input_file) as r:
        text = r.read()

    state = None
    if os.path.exists(output_file):
        state = State.load(output_file)
    else:
        paragraphs = [p for p in text.split("\n") if p.strip()]
        paragraphs = split_paragrahps(
            paragraphs,
            max_paragraph_length=max_paragraph_length,
            language=language
        )
        paragraphs = merge_paragraphs(
            paragraphs,
            min_paragraph_length=min_paragraph_length
        )
        state = State()
        state.paragraphs = paragraphs

    model_settings = ModelSettings(
        model_name=model_name,
        prompt_template="openai",
        generation_params=GenerationParams(temperature=0.3, repetition_penalty=1.25)
    )

    if not state.name:
        for window in gen_windows(
            state.paragraphs,
            input_tokens_limit=input_tokens_limit,
            model_settings=model_settings
        ):
            paragraphs = [p for _, p in window]
            name, language = extract_meta(paragraphs, model_settings)
            state.name = name
            state.language = language
            break

    for summary in summarize_paragraphs_by_windows(
        paragraphs=state.paragraphs,
        cached_summaries=state.l1_summaries,
        language=state.language,
        model_settings=model_settings,
        input_tokens_limit=input_tokens_limit,
        prompt="l1_summarize",
        num_sentences=10
    ):
        assert summary
        assert isinstance(summary, dict)
        state.l1_summaries.append(summary)
        state.save(output_file)

    state.l1_summaries = postprocess_l1(state.l1_summaries)
    state.save(output_file)

    l2_paragraphs = [[]]
    for point in state.l1_summaries:
        if "summary_point" in point:
            l2_paragraphs[-1].append(point["summary_point"])
            continue
        if "chapter_header" in point:
            l2_paragraphs.append([])
    l2_paragraphs = ["\n".join(p).strip() for p in l2_paragraphs if "\n".join(p).strip()]
    #for p in l2_paragraphs:
    #    print(p)
    #    print()
    #    print("=======")
    #    print()

    state.l2_summaries = []
    cached_l2_summaries_count = len(state.l2_summaries)
    for pnum, paragraph in enumerate(l2_paragraphs):
        if pnum < cached_l2_summaries_count:
            continue

        prev_summary = ""
        if state.l2_summaries:
            prev_summary = state.l2_summaries[-1]

        l2_summaries = []
        for summary in summarize_paragraphs_by_windows(
            paragraphs=[paragraph],
            language=state.language,
            model_settings=model_settings,
            input_tokens_limit=input_tokens_limit,
            prev_summary=prev_summary,
            prompt="l2_summarize",
            num_sentences=8
        ):
            l2_summaries.append(summary)

        print("END CHAPTER")
        state.l2_summaries.append("\n".join(l2_summaries))
        state.save(output_file)

    final_summary = summarize(
        paragraphs=state.l2_summaries,
        language=state.language,
        prompt="l2_summarize",
        num_sentences=8
    )


if __name__ == "__main__":
    fire.Fire(summarize_book)
