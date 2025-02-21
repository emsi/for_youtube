import os

from openai import OpenAI


def create_client():
    """Create OpenAI client with given API key and base URL"""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "xxx"),  # an invalid key to squash client errors
        base_url=os.getenv("OPENAI_BASE_URL", None),
    )


def ask(question: str, assistant_message: str | None = None):
    """Main entry point"""
    client = create_client()

    messages = [{"role": "user", "content": question}]

    if assistant_message:
        messages.append({"role": "assistant", "content": assistant_message})

    print(f"\n##########\nSending: {messages}\n##########\n")
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=messages,
        temperature=0,
        stream=True,
    )

    if assistant_message:
        print(assistant_message, end="")

    for chunk in response:
        print(chunk.choices[0].delta.content, end="")
    print()
