"""
Pattern 03: Function and Tool Calling with Cohere ClientV2
Demonstrates how to define tool schemas, let Cohere determine when to invoke a tool,
execute the Python function, and return the execution results back to the model for
the final grounded response.
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

co = cohere.ClientV2(api_key=api_key)

# 1. Define the tool schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_the_Language",
            "description": "Get the primary official language associated with a state or location in India.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "A state, city, or region name"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# 2. Implement the actual Python function
def get_the_Language(location: str) -> dict:
    loc_lower = location.lower()
    mapping = {
        "maharashtra": "Marathi",
        "karnataka": "Kannada",
        "tamil nadu": "Tamil",
        "telangana": "Telugu",
        "andhra pradesh": "Telugu",
        "west bengal": "Bengali",
        "gujarat": "Gujarati"
    }

    for state, lang in mapping.items():
        if state in loc_lower:
            return {"location": location, "language": lang}

    return {"location": location, "language": "Unknown - no mapping available in database"}

functions_map = {
    "get_the_Language": get_the_Language
}

query = "What is the primary official language spoken in Maharashtra, India?"
print("=" * 70)
print(f"User Query: {query}")
print("=" * 70)

messages = [
    {
        "role": "user",
        "content": query
    }
]

# 3. Ask Cohere to select and call a tool
print("\n[Step 1] Sending prompt to Cohere with tools registered...")
response = co.chat(
    model="command-a-plus-05-2026",
    messages=messages,
    tools=tools,
    temperature=0
)

# 4. Check if model requested tool execution
if response.message.tool_calls:
    print(f"\n[Step 2] Model selected tool calls: {len(response.message.tool_calls)}")
    messages.append(response.message)

    for tc in response.message.tool_calls:
        func_name = tc.function.name
        arguments = json.loads(tc.function.arguments)
        print(f"   Executing '{func_name}' with arguments: {arguments}")

        # Execute mapped local function
        result = functions_map[func_name](**arguments)
        print(f"   Result: {result}")

        # Append tool execution result
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result)
        })

    # 5. Ask Cohere to produce the final answer with tool context
    print("\n[Step 3] Submitting tool output back to Cohere for final response...")
    final_response = co.chat(
        model="command-a-plus-05-2026",
        messages=messages,
        tools=tools,
        temperature=0
    )

    text_parts = [
        item.text for item in final_response.message.content
        if getattr(item, "type", None) == "text"
    ]
    print("\n[Final Answer]:")
    print("\n".join(text_parts))
else:
    print("\nNo tool calls requested. Direct text response:")
    for item in response.message.content:
        if getattr(item, "type", None) == "text":
            print(item.text)
