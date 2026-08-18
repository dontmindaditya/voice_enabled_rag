import asyncio
import os
import time
from dotenv import load_dotenv
load_dotenv()
from groq import AsyncGroq
from src.rag.retriever import LanceRetriever

async def main():
    client = AsyncGroq()
    retriever = LanceRetriever()
    query = "What is the capital of India?"
    ret = retriever.retrieve(query)
    
    print("--- Running queries with groq/compound-mini ---")
    for i in range(5):
        t0 = time.perf_counter()
        response = await client.chat.completions.create(
            model="groq/compound-mini",
            messages=[
                {"role": "system", "content": "You are a low-latency factual system. Answer the query in 5-10 words using ONLY the Context. If unknown, say: 'I cannot answer based on the dataset.'"},
                {"role": "user", "content": f"Context: {ret['context']}\nQ: {query}\nA:"}
            ],
            temperature=0.0,
            max_tokens=15,
            stop=["\n", "Context:"]
        )
        latency = (time.perf_counter() - t0) * 1000
        ans = response.choices[0].message.content.strip()
        print(f"Run {i+1} Answer: {repr(ans)} (Latency: {latency:.2f} ms)")
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())
