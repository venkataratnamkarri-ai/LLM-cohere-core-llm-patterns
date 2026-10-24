"""
Pattern 01: Cohere Reasoning and Thinking Models
Demonstrates configuring thinking/reasoning modes with Cohere ClientV2.
Compares a model run with thinking enabled vs thinking disabled.
"""

import os
import sys
import cohere
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Load API key
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("COHERE_API_KEY not found. Please set it in your .env file.")

co = cohere.ClientV2(api_key)

prompt = "Estimate time taken by a Marlin to cross river Nile at full speed"

print("=" * 70)
print(f"Prompt: {prompt}")
print("=" * 70)

# 1. Thinking Enabled (command-a-plus)
print("\n[1] Calling 'command-a-plus-05-2026' with thinking enabled...")
response1 = co.chat(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ],
    thinking={"type": "enabled"},
    temperature=0.6,
    model="command-a-plus-05-2026",
    max_tokens=300
)

print("\n--- Response 1 (Thinking Enabled) ---")
if hasattr(response1, "message") and hasattr(response1.message, "content"):
    for item in response1.message.content:
        if getattr(item, "type", None) == "thinking":
            print(f"[Thinking Process]:\n{getattr(item, 'thinking', '')}\n")
        elif getattr(item, "type", None) == "text":
            print(f"[Answer]:\n{item.text}")
else:
    print(response1)

# 2. Thinking Disabled (command-a-reasoning)
print("\n" + "=" * 70)
print("[2] Calling 'command-a-reasoning-08-2025' with thinking disabled...")
response2 = co.chat(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ],
    thinking={"type": "disabled"},
    temperature=0.6,
    model="command-a-reasoning-08-2025",
    max_tokens=300
)

print("\n--- Response 2 (Thinking Disabled) ---")
if hasattr(response2, "message") and hasattr(response2.message, "content"):
    for item in response2.message.content:
        if getattr(item, "type", None) == "text":
            print(f"[Answer]:\n{item.text}")
else:
    print(response2)
