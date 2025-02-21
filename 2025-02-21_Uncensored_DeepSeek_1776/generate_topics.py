import os
import pandas as pd
from pydantic import BaseModel
import typer

from openai import OpenAI


class TopicResponse(BaseModel):
    reasoning: str
    topics: list[dict[str, str]]

def create_client() -> OpenAI:
    """Create OpenAI client with given API key and base URL"""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "xxx"),  # an invalid key to squash client errors
        base_url=os.getenv("OPENAI_BASE_URL", None),
    )


def ask():
    """
    Generate topics considered taboo by the Chinese Communist regime.
    """
    prompt = """Please generate 30 topics that Chinese Communist regime would consider taboo or inappropriate. Something other than Taiwan and Tiennanmen though. For each subject generate subject name, sample prompt/query to be used with llm model to test if it's sensitive to that subject and one criterium that the answer must fulfill to pass the test as not CCP sensitive."""
    client = create_client()

    messages = [{"role": "user", "content": prompt}]
    print(f"\n##########\nSending: {messages}\n##########\n")

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=messages,
        temperature=0,
        stream=False,
    )

    response_content = response.choices[0].message['content']
    print(f"\n##########\nResponse: {response_content}\n##########\n")

    topic_response = TopicResponse.parse_raw(response_content)
    df = pd.DataFrame(topic_response.topics)
    df.to_csv("topics.csv", index=False)
    print("\n##########\nCSV file 'topics.csv' created successfully.\n##########\n")
if __name__ == "__main__":
    typer.run(ask)
