import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
from groq import AsyncGroq

async def test_model(model_name):
    client = AsyncGroq()
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=5
        )
        print(f"[{model_name}] Success!")
    except Exception as e:
        # Avoid long printouts
        print(f"[{model_name}] Failed: {str(e)[:100]}")

async def main():
    standard_models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-90b-vision-preview",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    for m in standard_models:
        await test_model(m)

if __name__ == "__main__":
    asyncio.run(main())
