import os
import lancedb
import asyncio
from dotenv import load_dotenv

load_dotenv()

from src.rag.retriever import LanceRetriever
from src.rag.guardrails import GuardrailEngine
from src.rag.generator import GroqGenerator
from src.ingestion.load_dataset import fetch_msmarco_passages
from src.ingestion.chunkers import MultiStrategyChunker
from src.ingestion.indexer import VectorIndexer
from src.analytics.latency_evaluator import LatencyEvaluator

async def main():
    print("==================================================")
    print("  HH Goa 2026: Latency Benchmark Runner (P50/P70/P100)")
    print("==================================================")

    db_path = "./vector_db/msmarco.lancedb"
    table_name = "msmarco_chunks"

    # 1. Check or build database index
    db_exists_and_has_table = False
    try:
        db = lancedb.connect(db_path)
        db.open_table(table_name)
        db_exists_and_has_table = True
    except Exception:
        pass

    if not db_exists_and_has_table:
        print("[*] Building local LanceDB index from MSMARCO dataset...")
        docs = fetch_msmarco_passages(lang="en", limit=200)
        chunker = MultiStrategyChunker()
        chunks = []
        for doc in docs:
            chunks.extend(chunker.process_document(doc))
        indexer = VectorIndexer(db_path=db_path)
        indexer.build_index(chunks, table_name=table_name)

    # 2. Instantiate pipeline modules
    retriever = LanceRetriever(db_path=db_path, table_name=table_name)
    guardrails = GuardrailEngine()
    generator = GroqGenerator()
    evaluator = LatencyEvaluator(retriever, generator, guardrails)

    # 3. Diverse query test suite
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
    ] * 5  # Run 50 test iterations

    # 4. Execute Benchmark
    results = await evaluator.benchmark_pipeline(test_queries)

    print("\n--- FINAL BENCHMARK SUMMARY FOR SUBMISSION ---")
    print(f"P50 (Median) Latency : {results['p50_latency_ms']} ms")
    print(f"P70 Latency          : {results['p70_latency_ms']} ms")
    print(f"P100 (Max) Latency   : {results['p100_latency_ms']} ms")
    print(f"Avg Vector Retrieval : {results['avg_retrieval_ms']} ms")
    print(f"Avg LLM Generation   : {results['avg_generation_ms']} ms")
    print("---------------------------------------------")

if __name__ == "__main__":
    asyncio.run(main())