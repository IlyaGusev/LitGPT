import json
from openai_wrapper import openai_completion, encode_prompt, DEFAULT_MODEL


DEFAULT_SYSTEM_PROMPT = "You are a helpful and creative assistant for writing novel."


def novel_completion(
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
):
    messages=[{
        "role": "system",
        "content": "You are a helpful and creative assistant for writing novel."
    }, {
        "role": "user",
        "content": prompt,
    }]
    return openai_completion(messages, model_name=model_name)


def parse_json_output(output):
    start_index = output.find("{")
    end_index = output.rfind("}")
    text = output[start_index:end_index+1]
    text = text.strip()
    record = json.loads(text)
    return record


def get_init(description: str, novel_type: str):
    if description == "":
        description = "A novel about anything"
    if novel_type == "":
        novel_type = "Science Fiction"
    init_prompt = encode_prompt("prompts/init.jinja", description=description, novel_type=novel_type)
    response = novel_completion(init_prompt)
    print(response)
    output = parse_json_output(response)
    return output
