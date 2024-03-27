ALPACA_TEMPLATE = """### System:
{system_message}
### User:
{user_message}
### Assistant:
"""


CHATML_TEMPLATE = """<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""


SAIGA_TEMPLATE = """<s>system
{system_message}
<s>user
{user_message}</s>
<s>bot
"""


PROMPT_TEMPLATES = {
    "alpaca": ALPACA_TEMPLATE,
    "saiga": SAIGA_TEMPLATE,
    "chatml": CHATML_TEMPLATE,
    "openai": "openai",
    "anthropic": "anthropic",
    "custom": "Edit here",
}


def format_template(messages, template):
    if template in PROMPT_TEMPLATES:
        template = PROMPT_TEMPLATES[template]

    system_message = ""
    user_message = ""
    for message in messages:
        if message["role"] == "system":
            system_message = message["content"]
        elif message["role"] == "user":
            user_message = message["content"]
    assert user_message
    return template.format(system_message=system_message, user_message=user_message)


PROMPT_TEMPLATE_LIST = list(PROMPT_TEMPLATES.keys())
DEFAULT_PROMPT_TEMPLATE_NAME = "openai"
