ALPACA_TEMPLATE = """### System:
    {system_message}
### User:
    {user_message}
### Assistant:"""


def format_alpaca(messages):
    system_message = ""
    user_message = ""
    for message in messages:
        if message["role"] == "system":
            system_message = message["content"]
        elif message["role"] == "user":
            user_message = message["content"]
    assert user_message
    return ALPACA_TEMPLATE.format(
        system_message=system_message,
        user_message=user_message
    )


SAIGA_TEMPLATE = """<s>system
Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.</s>
<s>user
{user_message}</s>
<s>bot
"""


def format_saiga(messages):
    user_message = ""
    for message in messages:
        if message["role"] == "user":
            user_message = message["content"]
    assert user_message
    return SAIGA_TEMPLATE.format(
        user_message=user_message
    )


PROMPT_TEMPLATES = {
    "alpaca": format_alpaca,
    "saiga": format_saiga
}

PROMPT_TEMPLATE_LIST = ["openai"] + list(PROMPT_TEMPLATES.keys())

DEFAULT_PROMPT_TEMPLATE_NAME = "openai"
