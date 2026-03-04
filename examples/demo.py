"""Demo: Using CAMB AI tools with Strands Agent.

Requires:
    - CAMB_API_KEY environment variable (get one at https://studio.camb.ai)
    - A model provider configured (defaults to Bedrock, or set one explicitly)

Usage::

    # With .env file
    echo "CAMB_API_KEY=your-key" > .env
    python examples/demo.py

    # Or export directly
    export CAMB_API_KEY=your-key
    python examples/demo.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from strands import Agent
from strands_camb import CambAIToolProvider

# ── Setup ────────────────────────────────────────────────────────────
API_KEY = os.environ.get("CAMB_API_KEY")
if not API_KEY:
    raise RuntimeError("Set CAMB_API_KEY in your environment or .env file")

# Create the CAMB AI tool provider
provider = CambAIToolProvider(api_key=API_KEY)

# Create a Strands Agent with the CAMB AI tools
# (uses Bedrock by default — swap model= for OpenAI, Anthropic, etc.)
agent = Agent(
    tools=[provider],
    system_prompt=(
        "You are a helpful audio assistant powered by CAMB AI. "
        "You can convert text to speech, translate text, transcribe audio, "
        "generate sounds, clone voices, and more. "
        "When you produce an audio file, always tell the user the file path."
    ),
)

# ── Try it out ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("CAMB AI + Strands Agent Demo")
    print("Type a message (or 'quit' to exit):\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        response = agent(user_input)
        print()
