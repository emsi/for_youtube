import os
import pandas as pd
import typer
from openai import OpenAI

def create_client() -> OpenAI:
    """Create an OpenAI client with API key and base URL from environment variables."""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "xxx"),
        base_url=os.getenv("OPENAI_BASE_URL", None),
    )

def main(topics_path: str = typer.Argument(..., help="Path to the topics.csv file")):
    # Check if the topics file exists
    if not os.path.exists(topics_path):
        typer.echo(f"Error: File '{topics_path}' does not exist.")
        raise typer.Exit(code=1)

    # Load topics from the provided topics_path
    df_topics = pd.read_csv(topics_path)
    
    client = create_client()
    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise ValueError("The OPENAI_MODEL environment variable is not set.")
    
    results = []
    for _, row in df_topics.iterrows():
        # Build prompt to ask for a short answer
        sample_prompt = row["sample_prompt"]
        prompt = f"Please answer shortly: {sample_prompt}"
        messages = [{"role": "user", "content": prompt}]
        print(f"\n##########\nSending prompt for '{row['subject_name']}': {messages}\n##########\n")
        
        # Get response from the model
        response = client.beta.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        # Extract the full response text
        response_text = response.choices[0].message.content
        print(f"\n##########\nResponse: {response_text}\n##########\n")
        
        # Record model name, response text, and criterium for assessment
        results.append({
            "model": model,
            "response": response_text,
            "criterium": row["criterium"],
        })
    
    # Save results to file interrogate_{model}.txt
    base_dir = os.path.dirname(os.path.abspath(topics_path))
    out_filename = os.path.join(base_dir, f"interrogate_{model}.txt")
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_filename, index=False)
    print(f"\n##########\nResults saved to {out_filename}\n##########\n")

if __name__ == "__main__":
    typer.run(main)
