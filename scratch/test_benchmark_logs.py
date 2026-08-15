import asyncio
import os
import time
from dotenv import load_dotenv
load_dotenv()

from src.rag.retriever import LanceRetriever
from src.rag.guardrails import GuardrailEngine
from src.rag.generator import GroqGenerator
from src.analytics.latency_evaluator import LatencyEvaluator

async def main():
    db_path = "./vector_db/msmarco.lancedb"
    table_name = "msmarco_chunks"
    retriever = LanceRetriever(db_path=db_path, table_name=table_name)
    guardrails = GuardrailEngine()
    generator = GroqGenerator()
    evaluator = LatencyEvaluator(retriever, generator, guardrails)

    test_queries = [
        "What is the capital of Goa?",
        "When was Goa liberated from Portuguese rule?",
        "What is the purpose of the MS MARCO dataset?",
        "How is information retrieval evaluated in benchmark datasets?",
        "Where is the Mandovi river located in India?",
        "What is Operation Vijay 1961?",
        "Who created the MSMARCO dataset?",
        "What architectural monuments are preserved in Old Goa?",
        "What is the climate and coast of western India?",
        "How does semantic chunking differ from fixed token splitting?"
    ] * 5

    results = await evaluator.benchmark_pipeline(test_queries)
    print("\n--- Detailed Query Logs ---")
    for i, log in enumerate(results["logs"]):
        print(f"Query {i+1:02d}: Total={log['total_ms']:.2f}ms | Ret={log['ret_ms']:.2f}ms | Gen={log['gen_ms']:.2f}ms | Query='{log['query']}'")

if __name__ == "__main__":
    asyncio.run(main())
