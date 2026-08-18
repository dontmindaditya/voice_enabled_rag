import asyncio
import os
import time
from dotenv import load_dotenv
load_dotenv()
from groq import AsyncGroq

async def test_model(model_name):
    client = AsyncGroq()
    try:
        t0 = time.perf_counter()
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Who was the first prime minister of India?"}
            ],
            max_tokens=30
        )
        latency = (time.perf_counter() - t0) * 1000
        print(f"[{model_name}] Success: '{response.choices[0].message.content.strip()}' (Latency: {latency:.2f} ms)")
    except Exception as e:
        print(f"[{model_name}] Error: {str(e)}")

async def main():
    await test_model("canopylabs/orpheus-v1-english")

if __name__ == "__main__":
    asyncio.run(main())
