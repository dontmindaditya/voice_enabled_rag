import os
import time
from groq import AsyncGroq
from typing import Dict, Any

class GroqGenerator:
    """
    Sub-80ms generation harness utilizing Groq LPU free tier (llama-3.1-8b-instant).
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.client = AsyncGroq(api_key=self.api_key)
        self.model = "llama-3.1-8b-instant"

    async def generate_grounded_answer(self, query: str, context: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        
        system_instruction = (
            "You are a helpful and factual RAG assistant. "
            "Answer the user's question directly and concisely (1-2 sentences) using the provided Context. "
            "You may infer answers if the context directly addresses the core entity or question. "
            "Only say 'I cannot answer based on the provided dataset.' if the context is completely unrelated."
        )

        user_content = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0,
                max_tokens=35
            )
            answer = response.choices[0].message.content.strip()
            latency = (time.perf_counter() - t0) * 1000
            
            return {
                "answer": answer,
                "latency_ms": round(latency, 2)
            }
        except Exception as e:
            return {
                "answer": f"Generation error: {str(e)}",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)
            }