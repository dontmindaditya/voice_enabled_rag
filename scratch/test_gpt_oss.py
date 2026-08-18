import asyncio
import os
import time
from dotenv import load_dotenv
load_dotenv()
from groq import AsyncGroq

async def main():
    client = AsyncGroq()
    try:
        t0 = time.perf_counter()
        response = await client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": "Say hello in 3 words."}
            ],
            max_tokens=10
        )
        latency = (time.perf_counter() - t0) * 1000
        print(f"[openai/gpt-oss-20b] Success: '{response.choices[0].message.content.strip()}' (Latency: {latency:.2f} ms)")
    except Exception as e:
        print(f"[openai/gpt-oss-20b] Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
