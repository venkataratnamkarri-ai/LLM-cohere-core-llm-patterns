"""
Pattern 06: OpenAI Chat Baseline Comparison
Demonstrates calling the OpenAI API for chat responses as a baseline comparison
alongside Cohere's Command models.
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# Ensure UTF-8 output on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("Warning: OPENAI_API_KEY not found in .env. Skipping execution.")
    print("Set OPENAI_API_KEY in .env to run this example.")
    sys.exit(0)

client = OpenAI(api_key=openai_api_key)

prompt = "A cloud typically weighs around a million tons, yet it floats, why?"

print("=" * 70)
print(f"Prompt: {prompt}")
print("=" * 70)

model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
print(f"\nCalling OpenAI model '{model_name}'...")

try:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=250,
        temperature=0.7
    )

    answer = response.choices[0].message.content
    print("\n--- OpenAI Response ---")
    print(answer)

except Exception as err:
    print(f"\nOpenAI API call failed: {err}")
