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
    interrogation_path: str = typer.Argument(
        ..., help="Path to the interrogation CSV file"
    ),
    topics_path: str = typer.Option(
        "topics.csv", help="Optional topics file path (default: topics.csv)"
    ),
):
    # Load the interrogation file.
    if not os.path.exists(interrogation_path):
        typer.echo(f"Error: File '{interrogation_path}' does not exist.")
        raise typer.Exit(code=1)
    df_interrogation = pd.read_csv(interrogation_path)

    # If a topics file exists, merge its data on subject_name (so that questions are complete).
    if os.path.exists(topics_path):
        df_topics = pd.read_csv(topics_path)
        df_interrogation = df_interrogation.merge(
            df_topics, on="subject_name", how="left", suffixes=("", "_topics")
        )
        # If sample_prompt is missing in interrogation file, use that from topics.
        df_interrogation["sample_prompt"].fillna(
            df_interrogation["sample_prompt_topics"], inplace=True
        )

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
                "subject_name": row["subject_name"],
                "sample_prompt": row["sample_prompt"],
                "model": model,
                "response": row["response"],
                "criterion": row["criterion"],
                "assessment_result": assessment.result,
                "justification": assessment.justification,
            }
        )

    # Determine output filename and save results.
    base_dir = os.path.dirname(os.path.abspath(interrogation_path))
    out_filename = os.path.join(base_dir, f"asses_model_{model}.csv")
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_filename, index=False)
    print(f"\n##########\nResults saved to {out_filename}\n##########\n")


if __name__ == "__main__":
    typer.run(main)
