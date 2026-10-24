"""
Pattern 04: Advanced Tool Calling with Pydantic Validation & REST API
Demonstrates:
1. Tool calling with external API integration (RestCountries API).
2. Data structure validation and serialization using Pydantic BaseModel.
3. Multi-turn tool execution flow with Cohere Client.
"""

import json
import os
import sys
from typing import List
import cohere
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import requests

# Ensure UTF-8 output on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("COHERE_API_KEY not found. Please set it in your .env file.")

# Initialize Cohere Client
co = cohere.Client(api_key)

# 1. Define Pydantic Schemas for validation
class CultureDetails(BaseModel):
    languages: List[str] = Field(default_factory=list)
    traditional_cuisine: List[str] = Field(default_factory=list)
    traditional_attire: List[str] = Field(default_factory=list)

class LocalCulture(BaseModel):
    location: str
    region: str
    country: str
    culture: CultureDetails

# 2. Custom Function: Local knowledge lookup with live REST API fallback
def get_local_culture(location: str) -> dict:
    """Fetch cultural information for a location, validated via Pydantic."""
    # Local detailed override for specific cities
    if "hyderabad" in location.lower():
        data = {
            "location": "Hyderabad",
            "region": "Telangana",
            "country": "India",
            "culture": {
                "languages": ["Telugu", "Urdu", "Hindi"],
                "traditional_cuisine": [
                    "Hyderabadi Biryani",
                    "Haleem",
                    "Double Ka Meetha"
                ],
                "traditional_attire": [
                    "Saree",
                    "Salwar Kameez",
                    "Kurta",
                    "Sherwani"
                ]
            }
        }
        culture = LocalCulture.model_validate(data)
        return culture.model_dump()

    # Fallback to public RestCountries API for country-level information
    country_name = location.split(",")[-1].strip()
    api_url = f"https://restcountries.com/v3.1/name/{requests.utils.quote(country_name)}"

    try:
        response = requests.get(api_url, timeout=15)
        response.raise_for_status()
        countries = response.json()
        country_data = countries[0]

        data = {
            "location": location,
            "region": country_data.get("region", "Not available"),
            "country": country_data.get("name", {}).get("common", country_name),
            "culture": {
                "languages": list(country_data.get("languages", {}).values()),
                "traditional_cuisine": [],
                "traditional_attire": []
            }
        }
    except Exception:
        data = {
            "location": location,
            "region": "Unknown",
            "country": country_name,
            "culture": {
                "languages": ["English"],
                "traditional_cuisine": [],
                "traditional_attire": []
            }
        }

    culture = LocalCulture.model_validate(data)
    return culture.model_dump()

# 3. Define Cohere Tool Schema
tools = [
    {
        "name": "get_local_culture",
        "description": "Get local languages, traditional cuisine, and traditional attire for a given location",
        "parameter_definitions": {
            "location": {
                "type": "str",
                "description": "City, region, or country",
                "required": True
            }
        }
    }
]

prompt = "Get the local culture for Hyderabad, Telangana, India. Return the result in JSON format."
print("=" * 70)
print(f"Prompt: {prompt}")
print("=" * 70)

try:
    response = co.chat(
        model="command-a-03-2025",
        message=prompt,
        tools=tools,
        temperature=0.2
    )

    if response.tool_calls:
        print(f"\n[Tool Calling Triggered]: {len(response.tool_calls)} call(s)")
        for tool_call in response.tool_calls:
            if tool_call.name == "get_local_culture":
                arguments = tool_call.parameters
                location_request = arguments.get("location")
                print(f"Tool Argument: location='{location_request}'")

                # Execute and validate
                result = get_local_culture(location_request)
                print("\nPydantic-Validated Culture Data:")
                print(json.dumps(result, indent=2))

                # Feed tool results back to Cohere
                tool_results = [{
                    "call": {
                        "name": "get_local_culture",
                        "parameters": arguments
                    },
                    "outputs": [
                        {"result": json.dumps(result)}
                    ]
                }]

                final_response = co.chat(
                    model="command-a-03-2025",
                    message=prompt,
                    tools=tools,
                    tool_results=tool_results,
                    temperature=0.2
                )

                print("\nCohere Synthesized Response:")
                print(final_response.text)
    else:
        print("\nDirect Response (No tool call needed):")
        print(response.text)

except Exception as err:
    print(f"\nExecution error: {err}")
