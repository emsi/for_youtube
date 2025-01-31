# Deep Thinking Agent

This script simulates deep thinking akin to reasoning models like r1 or o1 but with settable number of rounds.
It works with OpenAI compatible blackends like Ollama or DeepSeek's API but not with official OpenAI as it does not allow for the assistent message continuation trick.

## Deep Thinking Mechanism

The assistant employs a chain-of-thought prompting strategy that leverages LLM message continuation capabilities through 3 key phases:

```
[User Question]
  │
  ▼
[System: Start thinking tag]
  │
  ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Initial Think │ → │ Thought Nudge │ → │ Final Think   │
│ <thinking>    │   │ (But wait...) │   │ </thinking>   │
└───────────────┘   └───────────────┘   └───────────────┘
  │                     │                     │
  └─▶ Model Reasoning  ◀─┴─▶ Deeper Analysis ◀─┴─▶ Final Synthesis
```

1. **Assistants Message Continuation**  
We prime the model by a specialy crafted system message and by starting an assistant message with `<thinking>`. This creates "continuation tension" making the model eager to keep expanding its analysis.

2. **Thought Injections**  
Whenever model wants to finish the thinking section and emit `</thinkin>` tag,  we inject phrases like:
```python
["But wait...", "Alternatively...", "Taking a step back", "Alternatively, perhaps", *25+ variations*]
``` 
These act as cognitive nudges, forcing the model to:
- Re-examine assumptions
- Consider opposing viewpoints
- Explore edge cases
- Surface implicit constraints

This is repeated for --rounds number of times (default 3).

3. **Controlled Termination**  
Final `</thinking>` tag signals transition from analysis to answer synthesis. Then model automatically organizes its layered reasoning into structured output.

**Why This Works**  
The technique exploits LLMs' pattern completion instincts while maintaining controlled divergence. By strategically inserting "but..." phrases, we simulate the cognitive process of an expert asking himself:
1. "Have I considered X?"
2. "What if Y is different?"
3. "Does this hold under Z conditions?"

**Key Features**
- 🌀 Configurable "thinking depth" via `--rounds`
- 🤖 Compatible with any OpenAI API-compatible endpoint
- 🧠 Avoid official OpenAI due to restrictions in its implementation
- 🔄 Real-time streaming with thought injection visualization
- 🎲 Randomly selected challenge phrases prevent pattern overfitting

> 💡 Pro Tip: Combine with local models via Ollama for long free thinking loops!

# Usage
```
Usage: deep_thinking.py [OPTIONS] [QUESTION]

  Deep Thinking Assistant CLI

Options:
  --rounds INTEGER     Number of thinking rounds.
  --model TEXT         Model to use for the Deep Thinking Assistant, e.g.
                       gpt-4o-mini, deepseek-chat, qwen, etc.
  --baseline_url TEXT  Base URL for the OpenAI SDK compatible API.
  --api_key TEXT       API key for the OpenAI SDK.
  --help               Show this message and exit.
```
