"""
Pattern 02: Structured JSON Outputs with JSON Schema
Demonstrates enforcing strict JSON output from Cohere models using response_format
with a defined JSON schema (properties, types, and required fields).
"""

import json
import os
import sys
import cohere
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("COHERE_API_KEY not found. Please set it in your .env file.")

co = cohere.ClientV2(api_key)

prompt = """
Steps:
1. Who is the writer of the book "Gitanjali"?
2. Did the writer receive any Nobel prize for that?
3. Which year was it?
4. Name few best selling books by this writer.
"""

print("=" * 70)
print(f"Prompt: {prompt.strip()}")
print("=" * 70)

# Call Cohere v2 Chat API with response_format json_object and json_schema
response = co.chat(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ],
    thinking={"type": "enabled"},
    temperature=0.3,
    model="command-a-plus-05-2026",
    response_format={
        "type": "json_object",
        "json_schema": {
            "type": "object",
            "properties": {
                "writer": {
                    "type": "string",
                    "description": "Full name of the author"
                },
                "prize": {
                    "type": "boolean",
                    "description": "True if Nobel prize was received, else False"
                },
                "year": {
                    "type": "integer",
                    "description": "Year of Nobel prize award"
                },
                "best_selling_books": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of prominent books by the author"
                }
            },
            "required": [
                "writer",
                "prize",
                "year",
                "best_selling_books"
            ]
        }
    }
)

raw_output = ""
if hasattr(response, "message") and hasattr(response.message, "content"):
    for item in response.message.content:
        if getattr(item, "type", None) == "text" and item.text:
            raw_output += item.text.strip()

print("\nRaw Model JSON Output:")
print(raw_output)

# Validate and format with Python's json library
try:
    parsed_json = json.loads(raw_output)
    print("\nParsed and Formatted JSON Object:")
    print(json.dumps(parsed_json, indent=2))
except json.JSONDecodeError as err:
    print(f"\nFailed to parse model output as valid JSON: {err}")
