ALPACA_TEMPLATE = """### System:
    {system_message}
### User:
    {user_message}
### Assistant:"""


def alpaca_completion(func, messages, **kwargs):
    system_message = ""
    user_message = ""
    for message in messages:
        if message["role"] == "system":
            system_message = message["content"]
        elif message["role"] == "user":
            user_message = message["content"]
    assert user_message
    prompt = ALPACA_TEMPLATE.format(
        system_message=system_message,
        user_message=user_message
    )
    return func(
        prompt=prompt,
        **kwargs
    )
