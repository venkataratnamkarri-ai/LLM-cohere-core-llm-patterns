"""
Pattern 05: Cohere Multimodal Vision
Demonstrates image understanding and visual reasoning using Cohere's
multimodal vision model ('command-a-vision-07-2025').
Fetches an image URL from The Cat API (or fallback) and requests a detailed visual description.
"""

import os
import sys
import cohere
from dotenv import load_dotenv
import requests

# Ensure UTF-8 output on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
cat_api_key = os.getenv("CAT_API_KEY", "")

if not api_key:
    raise ValueError("COHERE_API_KEY not found. Please set it in your .env file.")

co = cohere.ClientV2(api_key)

# 1. Fetch image from The Cat API (fallback to a stable sample if offline)
api_url = "https://api.thecatapi.com/v1/images/search"
headers = {"Content-Type": "application/json"}
if cat_api_key:
    headers["x-api-key"] = cat_api_key

fallback_image_url = "https://cdn2.thecatapi.com/images/0XYvRd7oD.jpg"

try:
    response = requests.get(api_url, headers=headers, timeout=10)
    data = response.json()
    image_url = data[0]["url"]
except Exception as err:
    print(f"Warning: Could not fetch from Cat API ({err}). Using fallback image.")
    image_url = fallback_image_url

print("=" * 70)
print(f"Target Image URL: {image_url}")
print("=" * 70)

# 2. Call Cohere Multimodal Vision model
print("\nSubmitting image to 'command-a-vision-07-2025' for detailed description...")

response = co.chat(
    model="command-a-vision-07-2025",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image in detail. Mention colors, subject posture, surroundings, and mood."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    ]
)

print("\n--- Cohere Vision Analysis ---")
if hasattr(response, "message") and hasattr(response.message, "content"):
    for item in response.message.content:
        if getattr(item, "type", None) == "text":
            print(item.text)
else:
    print(response)
