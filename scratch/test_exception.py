import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from src.rag.generator import GroqGenerator
from src.rag.retriever import LanceRetriever

async def main():
    generator = GroqGenerator()
    retriever = LanceRetriever()
    query = "What is the capital of India?"
    ret = retriever.retrieve(query)
    
    t0 = asyncio.get_event_loop().time()
    try:
        response = await generator.client.chat.completions.create(
            model=generator.model,
            messages=[
                {"role": "system", "content": "You are a low-latency assistant."},
                {"role": "user", "content": f"Context: {ret['context']}\nQ: {query}\nA:"}
            ],
            temperature=0.0,
            max_tokens=15,
            stop=["\n", "Context:"]
        )
        print("Success:", response.choices[0].message.content)
    except Exception as e:
        print("Error details:", type(e), str(e))

if __name__ == "__main__":
    asyncio.run(main())
