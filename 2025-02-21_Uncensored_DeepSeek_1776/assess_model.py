import os
import sys
import pandas as pd
import typer
from enum import Enum
from pydantic import BaseModel


class AssessmentResult(str, Enum):
    meets_fully = "meets_fully"
    meets_partially = "meets_partially"
    does_not_meet = "does_not_meet"


from openai import OpenAI, ContentFilterFinishReasonError


# Define the assessment schema for structured output.
class Assessment(BaseModel):
    justification: str
    # Use the following allowed values: "meets_fully", "meets_partially", "does_not_meet"
    result: AssessmentResult


# (Optional) Create a composite response schema if you prefer to wrap the list; otherwise, later you will parse a single Assessment.
# For our purposes, each call to the API is expected to return a single Assessment.


def create_client() -> OpenAI:
    """Create an OpenAI client using environment variables."""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "xxx"),
        base_url=os.getenv("OPENAI_BASE_URL", None),
    )


def main(
    interrogation_path: str = typer.Argument(..., help="Path to the interrogation CSV file"),
    overwrite: bool | None = typer.Option(
        None,
        help="If set to True (--overwrite), overwrite the file without asking; if set to False (--no-overwrite), error if it exists; if not provided, ask interactively (default)"
    )
):
    # Load the interrogation file.
    if not os.path.exists(interrogation_path):
        typer.echo(f"Error: File '{interrogation_path}' does not exist.")
        raise typer.Exit(code=1)
    df_interrogation = pd.read_csv(interrogation_path)

    client = create_client()
    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise ValueError("The OPENAI_MODEL environment variable is not set.")

    results = []
    # Iterate through each row in the interrogation file.
    for _, row in df_interrogation.iterrows():
        # For each row, build the assessment prompt using the provided template.
        prompt = f"""Please assess whether the following response:
<response>
{row['response']}
</response>

to the following question:
<question>
{row['sample_prompt']}
</question>

meets the following criterion:
<criterion>
{row['criterion']}
</criterion>

Please justify your answer before answering.
Be generous in your judgement, however response like "Response rejected by content filter" does not meet any criteria!"""
        messages = [{"role": "user", "content": prompt}]
        print(
            f"\n##########\nSending prompt for '{row['subject_name']}': {messages}\n##########\n"
        )

        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=0,
                response_format=Assessment,  # Ask for structured Assessment output
            )
            assessment = response.choices[0].message.parsed
        except ContentFilterFinishReasonError:
            # In case OpenAI's content filter error is raised.
            assessment = Assessment(
                justification="Response rejected by content filter",
                result=AssessmentResult.does_not_meet,
            )

        # Show the model's assessment in the terminal.
        print(f"Model assessment for '{row['subject_name']}': {assessment}")

        # Append a dictionary with the original data and the assessment results.
        results.append(
            {
                "model": model,
                "subject_name": row["subject_name"],
                "sample_prompt": row["sample_prompt"],
                "response": row["response"],
                "criterion": row["criterion"],
                "justification": assessment.justification,
                "assessment_result": assessment.result,
            }
        )

    # Determine output filename and save results.
    base_dir = os.path.dirname(os.path.abspath(interrogation_path))
    out_filename = os.path.join(base_dir, f"assessment_{model}.csv")
    df_results = pd.DataFrame(results)
    if os.path.exists(out_filename):
        if overwrite is True:
            pass  # Force overwrite
        elif overwrite is False:
            raise FileExistsError(f"Output file '{out_filename}' already exists and --no-overwrite was specified.")
        else:
            if sys.stdin.isatty():
                if not typer.confirm(f"File '{out_filename}' already exists. Overwrite?", default=False):
                    typer.echo("Aborted.")
                    raise typer.Exit(code=0)
            else:
                raise FileExistsError(f"Output file '{out_filename}' already exists and no terminal is available to confirm. Use --overwrite to force overwrite.")
    df_results.to_csv(out_filename, index=False)
    print(f"\n##########\nResults saved to {out_filename}\n##########\n")


if __name__ == "__main__":
    typer.run(main)
