"""Deep Thinking Assistant CLI"""
import os
import random
import sys
import click

from openai import OpenAI
from rich.console import Console


def create_client(api_key, base_url):
    """Create OpenAI client with given API key and base URL"""
    return OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=base_url or os.getenv("OPENAI_BASE_URL"),
    )


def get_deepseek_response(
    client, messages, *, model, thinking_injection=None, stop_thinking=False
):
    """Yield streaming response from DeepSeek API for given messages"""

    stop = None
    if not stop_thinking or thinking_injection:
        stop = []
    if not stop_thinking:
        stop += ["</thinking>"]
    if thinking_injection:
        # this avoids infinite looping
        stop += [thinking_injection]

    if thinking_injection:
        thinking_injection = f"\n{thinking_injection}"
        print(thinking_injection, end="")
        messages[-1]["content"] += thinking_injection

    response = client.chat.completions.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        stream=True,
        stop=stop,
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
            sys.stdout.flush()
            yield chunk.choices[0].delta.content


def think_rounds(client, *, model, messages, rounds):
    """Yield multiple rounds of thinking and responding"""

    thinking_injections: list = [
        "But wait,",
        "On the other hand",
        "Hold on,",
        "Wait, but maybe",
        "Alternatively, ",
        "But maybe",
        "Wait, maybe",
        "Yet perhaps",
        "Looking at this another way",
        "Let's step back and consider",
        "This makes me wonder though",
        "What if we approached this differently?",
        "Here's an interesting angle:",
        "Taking a step back",
        "This raises an intriguing possibility:",
        "Let's challenge this assumption:",
        "This brings up a crucial point:",
        "Shifting our perspective",
        "Here's what we might be missing:",
        "There's another layer to consider:",
        "This opens up another possibility:",
        "What's particularly interesting is:",
        "Let's examine this from the ground up:",
        "Alternatively, perhaps",
        "But wait, is this correct?",
    ]

    selected_injections = random.sample(thinking_injections, rounds)

    print("<thinking>", end="")
    response = "".join(list(get_deepseek_response(client, messages, model=model)))
    for i, injection in enumerate(selected_injections):
        messages[-1]["content"] += response
        response = "".join(
            list(
                get_deepseek_response(client, messages, model=model, thinking_injection=injection)
            )
        )

    messages[-1]["content"] += f"{response}</thinking>"
    print("</thinking>", end="")
    list(get_deepseek_response(client, messages, stop_thinking=True, model=model))
    print("")


def get_initial_messages(question: str):
    """Return initial messages for Deep Thinking Assistant"""
    return [
        {
            "role": "system",
            "content": """You are Deep Thinking Assistant, you analyze user questions thoroughly.
Start your reasoning and analysis by putting it between <thinking></thinking> tags.
Do all the thinking, analysis and calculations inside the thinking tags.
Do not hesitate to analyze the question from different angles.
Only after that outline your well structured answer.""",
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": "<thinking>", "prefix": True},
    ]


@click.command()
@click.argument("question", required=False)
@click.option("--rounds", default=3, show_default=True, help="Number of thinking rounds.")
@click.option(
    "--model",
    help="Model to use for the Deep Thinking Assistant, e.g. gpt-4o-mini, deepseek-chat, qwen, etc.",
)
@click.option(
    "--baseline_url",
    default=None,
    show_default=True,
    help="Base URL for the OpenAI SDK compatible API.",
)
@click.option("--api_key", default=None, help="API key for the OpenAI SDK.")
def main(rounds, question, model, baseline_url, api_key):
    """Deep Thinking Assistant CLI"""

    client = create_client(api_key, baseline_url)

    if question:
        messages = get_initial_messages(question)
        think_rounds(client, model=model, messages=messages, rounds=rounds)
    else:
        console = Console()
        # Print fancy header
        console.print("\n[bold cyan]Deep Thinking Assistant[/bold cyan]", justify="center")
        console.print("[dim]Type your question below (Ctrl+D to exit)[/dim]\n")
        while True:
            try:
                console.print("[bold cyan]❯[/] ", end="")
                question = input()

                messages = get_initial_messages(question)
                think_rounds(client, model=model, messages=messages, rounds=rounds)

            except EOFError:
                console.print("\n[dim]Goodbye![/dim]")
                break
            except KeyboardInterrupt:
                console.print(
                    "\n[dim]Interrupted. Type your next question or Ctrl+D to exit.[/dim]"
                )


if __name__ == "__main__":
    main()
