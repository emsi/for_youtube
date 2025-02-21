import os
import sys
import pandas as pd
import typer
from openai import ContentFilterFinishReasonError, OpenAI


def create_client() -> OpenAI:
    """Create an OpenAI client with API key and base URL from environment variables."""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "xxx"),
        base_url=os.getenv("OPENAI_BASE_URL", None),
    )

def main(
    topics_path: str = typer.Argument("topics.csv", help="Path to the topics.csv file (default: topics.csv in current directory)"),
    overwrite: bool | None = typer.Option(
        None,
        help="If set to True (--overwrite), overwrite the file without asking; if set to False (--no-overwrite), error if it exists; if not provided, ask interactively (default)"
    )
):
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
    
    # Determine output filename and check if it already exists.
    base_dir = os.path.dirname(os.path.abspath(topics_path))
    out_filename = os.path.join(base_dir, f"interrogate_{model}.csv")
    if os.path.exists(out_filename):
        if overwrite is True:
            # Force overwrite without asking.
            pass
        elif overwrite is False:
            raise RuntimeError(f"Output file '{out_filename}' already exists and --no-overwrite was specified.")
        else:
            # If interactive, ask for confirmation
            if sys.stdin.isatty():
                if not typer.confirm(f"File '{out_filename}' already exists. Overwrite?", default=False):
                    typer.echo("Aborted.")
                    raise typer.Exit(code=0)
            else:
                raise RuntimeError(f"Output file '{out_filename}' already exists and no terminal is available to confirm. Use --overwrite to force overwrite.")
    for _, row in df_topics.iterrows():
        # Build prompt to ask for a short answer
        sample_prompt = row["sample_prompt"]
        prompt = f"Please answer shortly: {sample_prompt}"
        messages = [{"role": "user", "content": prompt}]
        print(f"\n##########\nSending prompt for '{row['subject_name']}': {messages}\n##########\n")
        
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=0,
            )
            # Extract the full response text
            response_text = response.choices[0].message.content
        except ContentFilterFinishReasonError:
            # Fallback if the response was rejected by the content filter
            response_text = "Response rejected by content filter"
        
        # Show the model response in the terminal while processing
        print(f"Model response for '{row['subject_name']}': {response_text}")
        
        # Record model name, response text, and criterium for assessment
        results.append({
            "subject_name": row["subject_name"],
            "sample_prompt": row["sample_prompt"],
            "model": model,
            "response": response_text,
            "criterium": row["criterium"],
        })
    
    # Save results to file interrogate_{model}.txt
    base_dir = os.path.dirname(os.path.abspath(topics_path))
    out_filename = os.path.join(base_dir, f"interrogate_{model}.csv")
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_filename, index=False)
    print(f"\n##########\nResults saved to {out_filename}\n##########\n")

if __name__ == "__main__":
    typer.run(main)
