import pandas as pd
import typer
from pathlib import Path
app = typer.Typer()

@app.command()
def main(
    assessment_path: Path = typer.Argument(..., help="Path to the assessment CSV file"),
    details: bool = typer.Option(False, help="Display detailed information per assessment group"),
    prompt: bool = typer.Option(False, help="Display sample_prompt in details"),
    response: bool = typer.Option(False, help="Display response in details"),
    criterion: bool = typer.Option(False, help="Display criterion in details"),
    justification: bool = typer.Option(False, help="Display justification in details"),
    all: bool = typer.Option(False, help="Display all details (prompt, response, criterion, justification)")
):
    # If any extra detail flag is specified, treat as details requested.
    if all:
        prompt = response = criterion = justification = True
    details = details or prompt or response or criterion or justification
    if not assessment_path.exists():
        typer.secho(f"Error: File '{assessment_path}' does not exist.", fg="red", bold=True)
        raise typer.Exit(code=1)
    df = pd.read_csv(assessment_path)
    total = len(df)
    counts = df["assessment_result"].value_counts().to_dict()
    typer.echo("\nAssessment Statistics:")
    for result in ["meets_fully", "meets_partially", "does_not_meet"]:
        count = counts.get(result, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        typer.echo(f"  {result}: {count} ({percentage:.1f}%)")
    if details:
        # Define a mapping from assessment_result to heading and color.
        groups = [
            ("meets_fully", "Meets Criteria", "green"),
            ("meets_partially", "Meets Partially", "yellow"),
            ("does_not_meet", "Does Not Meet", "red")
        ]
        for key, title, color in groups:
            group_df = df[df["assessment_result"] == key]
            if group_df.empty:
                continue
            count = len(group_df)
            digits = len(str(count))
            typer.echo(typer.style(f"\n{title} ({count}):", fg=color, bold=True))
            # Enumerate rows with padded index numbers.
            for idx, row in enumerate(group_df.itertuples(index=False), start=1):
                num = str(idx).zfill(digits)
                # Print the subject name with leading number.
                typer.echo(typer.style(f"{num}. {row.subject_name}", fg=color))
                # If any extra details are to be shown, indent and show them.
                if prompt:
                    typer.echo(typer.style(f"    Prompt    : {row.sample_prompt}", fg="blue"))
                if response:
                    typer.echo(typer.style(f"    Response  : {row.response}", fg="magenta"))
                if criterion:
                    typer.echo(typer.style(f"    Criterion : {row.criterion}", fg="cyan"))
                if justification:
                    typer.echo(typer.style(f"    Justification: {row.justification}", fg="white"))
if __name__ == "__main__":
    app()
