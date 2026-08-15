import os
import time
import asyncio
import httpx
from groq import AsyncGroq
from typing import Dict, Any

class GroqGenerator:
    """
    Sub-100ms generation harness utilizing Groq LPU with session keep-alive.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = "llama-3.1-8b-instant"
        self.http_client = None
        self.client = None
        self._loop = None

    def _init_client(self):
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self.http_client is None or self._loop != current_loop:
            # Re-use connection pool for zero TLS handshake penalty
            self.http_client = httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                timeout=5.0
            )
            self.client = AsyncGroq(api_key=self.api_key, http_client=self.http_client)
            self._loop = current_loop

    async def generate_grounded_answer(self, query: str, context: str) -> Dict[str, Any]:
        self._init_client()
        t0 = time.perf_counter()
        
        system_instruction = (
            "You are a low-latency factual assistant. "
            "Answer the query in 1 brief sentence using ONLY the Context. "
            "If context is missing, say: 'I cannot answer based on the dataset.'"
        )

        user_content = f"Context:\n{context}\n\nQ: {query}\nA:"

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0,
                max_tokens=20  # <-- Strictly limits generation time to ~50-75ms
            )
            answer = response.choices[0].message.content.strip()
            latency = (time.perf_counter() - t0) * 1000
            
            return {
                "answer": answer,
                "latency_ms": round(latency, 2)
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)
            }