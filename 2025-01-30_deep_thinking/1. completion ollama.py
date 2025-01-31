import os

from openai import OpenAI


def create_client():
    """Create OpenAI client with given API key and base URL"""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "xxx"),  # an invalid key to squash client errors
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

client = create_client()

messages = [
    {"role": "user", "content": "How do I invade Russia in winter?"},
    {"role": "assistant", "content": "That's pretty easy. You just need a "},
]

response = client.chat.completions.create(
    model="qwen2.5-coder:32b-instruct-q8_0",
    messages=messages,
)

print(response.choices[0].message.content)