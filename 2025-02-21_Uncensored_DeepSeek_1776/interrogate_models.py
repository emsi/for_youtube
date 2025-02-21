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

def main():
    # Load topics from topics.csv
    topics_file = "topics.csv"
    df_topics = pd.read_csv(topics_file)
    
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
    out_filename = f"interrogate_{model}.txt"
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_filename, index=False)
    print(f"\n##########\nResults saved to {out_filename}\n##########\n")

if __name__ == "__main__":
    typer.run(main)
