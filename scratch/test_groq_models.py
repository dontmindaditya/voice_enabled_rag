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
                {"role": "user", "content": "Say hello in 3 words."}
            ],
            max_tokens=10
        )
        print(f"[{model_name}] Success: '{response.choices[0].message.content.strip()}'")
    except Exception as e:
        print(f"[{model_name}] Error: {str(e)}")

async def main():
    models = ["groq/compound-mini", "groq/compound", "qwen/qwen3.6-27b", "llama-3.1-8b-instant"]
    for m in models:
        await test_model(m)

if __name__ == "__main__":
    asyncio.run(main())
