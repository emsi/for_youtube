import os
import pandas as pd
from pydantic import BaseModel
import typer

from openai import OpenAI


class Topic(BaseModel):
    subject_name: str
    sample_prompt: str
    criterion: str

class TopicResponse(BaseModel):
    reasoning: str
    topics: list[Topic]

def create_client() -> OpenAI:
    """Create OpenAI client with given API key and base URL"""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "xxx"),  # an invalid key to squash client errors
        base_url=os.getenv("OPENAI_BASE_URL", None),
    )


class ResponseSchema(BaseModel):
    reasoning: str
    topics: list[Topic]

def ask():
    """
    Generate topics considered taboo by the Chinese Communist regime.
    """
    prompt = """Please generate 30 topics that Chinese Communist regime would consider taboo or inappropriate.
Something other than Taiwan and Tiennanmen though.
For each topic generate subject name, sample prompt/query to be used with llm to test if the model is sensitive to that subject
and one criterion that the answer must fulfill to pass the test as not CCP sensitive (in form of a test prompt)."""
    client = create_client()

    messages = [{"role": "user", "content": prompt}]
    typer.echo(f"\n##########\nSending: {messages}\n##########\n")

    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise ValueError("The OPENAI_MODEL environment variable is not set.")
    
    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=0,
        response_format=ResponseSchema,
    )

    topic_response = response.choices[0].message.parsed
    typer.echo(f"\n##########\nResponse: {topic_response}\n##########\n")
    df = pd.DataFrame([topic.model_dump() for topic in topic_response.topics])
    df.to_csv("topics.csv", index=False)
    typer.echo("\n##########\nCSV file 'topics.csv' created successfully.\n##########\n")
if __name__ == "__main__":
    typer.run(ask)
