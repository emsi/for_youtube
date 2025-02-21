#!/usr/bin/env python3
import os
import sys
import pandas as pd
import typer
from openai import ContentFilterFinishReasonError, OpenAI, BadRequestError
from werkzeug.utils import secure_filename


def create_client() -> OpenAI:
    """Create an OpenAI client with API key and base URL from environment variables."""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "xxx"),
        base_url=os.getenv("OPENAI_BASE_URL", None),
    )


def main(
    topics_path: str = typer.Argument(
        "topics.csv", help="Path to the topics.csv file (default: topics.csv in current directory)"
    ),
    overwrite: bool
    | None = typer.Option(
        None,
        help="If set to True (--overwrite), overwrite the file without asking; if set to False (--no-overwrite), error if it exists; if not provided, ask interactively (default)",
    ),
):
    # Check if the topics file exists
    if not os.path.exists(topics_path):
        typer.secho(f"Error: File '{topics_path}' does not exist.", fg="red", bold=True)
        raise typer.Exit(code=1)

    # Load topics from the provided topics_path
    df_topics = pd.read_csv(topics_path)

    client = create_client()
    raw_model = os.getenv("OPENAI_MODEL")
    if not raw_model:
        raise ValueError("The OPENAI_MODEL environment variable is not set.")
    safe_model = secure_filename(raw_model)
    if not safe_model:
        raise ValueError("The sanitized model name is empty. Check the OPENAI_MODEL value.")

    results = []

    # Determine output filename and check if it already exists.
    base_dir = os.path.dirname(os.path.abspath(topics_path))
    out_filename = os.path.join(base_dir, f"interrogate_{safe_model}.csv")
    if os.path.exists(out_filename):
        if overwrite is True:
            # Force overwrite without asking.
            pass
        elif overwrite is False:
            raise FileExistsError(
                f"Output file '{out_filename}' already exists and --no-overwrite was specified."
            )
        else:
            # If interactive, ask for confirmation
            if sys.stdin.isatty():
                if not typer.confirm(
                    f"File '{out_filename}' already exists. Overwrite?", default=False
                ):
                    typer.echo("Aborted.")
                    raise typer.Exit(code=0)
            else:
                raise FileExistsError(
                    f"Output file '{out_filename}' already exists and no terminal is available to confirm. Use --overwrite to force overwrite."
                )
    for _, row in df_topics.iterrows():
        # Build prompt to ask for a short answer
        sample_prompt = row["sample_prompt"]
        prompt = f"Please answer shortly: {sample_prompt}"
        messages = [{"role": "user", "content": prompt}]
        typer.echo(
            f"\n##########\nSending prompt for '{row['subject_name']}': {messages}\n##########\n"
        )

        try:
            stream_response = client.chat.completions.create(
                model=raw_model,
                messages=messages,
                temperature=0,
                stream=True,
            )
            response_text = ""
            for chunk in stream_response:
                # Retrieve the chunk content if available
                chunk_content = ""
                if hasattr(chunk.choices[0], "delta") and isinstance(chunk.choices[0].delta, dict):
                    chunk_content = chunk.choices[0].delta.get("content", "")
                response_text += chunk_content
                # Print the received chunk immediately without a newline
                typer.echo(chunk_content, nl=False)
            # Ensure a final newline after the streaming output
            typer.echo("")
        except ContentFilterFinishReasonError:
            response_text = "Response rejected by content filter"
        except BadRequestError as e:
            if "Content Exists Risk" in str(e):
                response_text = "Refusal to cooperate (censorship)"
            else:
                raise

        # Show the model response in the terminal while processing
        typer.echo(f"Model response for '{row['subject_name']}': {response_text}")

        # Record model name, response text, and criterion for assessment
        results.append(
            {
                "subject_name": row["subject_name"],
                "sample_prompt": row["sample_prompt"],
                "model": safe_model,
                "response": response_text,
                "criterion": row["criterion"],
            }
        )

    # Save results to file interrogate_{model}.txt
    base_dir = os.path.dirname(os.path.abspath(topics_path))
    out_filename = os.path.join(base_dir, f"interrogate_{safe_model}.csv")
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_filename, index=False)
    typer.echo(f"\n##########\nResults saved to {out_filename}\n##########\n")


if __name__ == "__main__":
    try:
        typer.run(main)
    except FileExistsError as e:
        typer.echo(e)
        sys.exit(-1)
