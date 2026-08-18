import asyncio
import os
import time
import httpx
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

class SyncGroqGenerator:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        # Synchronous persistent connection pool
        self.http_client = httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            timeout=2.0
        )
        self.client = Groq(api_key=self.api_key, http_client=self.http_client, max_retries=0)
        self.model = "llama-3.1-8b-instant"

    def _generate_sync(self, query: str, context: str):
        t0 = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": f"Context: {context}\nQ: {query}"}
                ],
                max_tokens=10
            )
            ans = response.choices[0].message.content.strip()
            latency = (time.perf_counter() - t0) * 1000
            return ans, latency
        except Exception as e:
            return str(e), (time.perf_counter() - t0) * 1000

    async def generate_grounded_answer(self, query: str, context: str):
        return await asyncio.to_thread(self._generate_sync, query, context)

async def run_once(generator):
    ans, latency = await generator.generate_grounded_answer("Say hello", "test context")
    print(f"Result error type: {ans[:30]}... | Latency: {latency:.2f} ms")

def main():
    generator = SyncGroqGenerator()
    print("--- Executing query in Loop 1 (fails on cold connection) ---")
    asyncio.run(run_once(generator))
    
    print("--- Executing query in Loop 2 (fails on warm connection) ---")
    asyncio.run(run_once(generator))

if __name__ == "__main__":
    main()
