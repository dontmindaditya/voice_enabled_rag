import os
import asyncio
os.environ["BENCHMARK_MODE"] = "true"

from src.rag.retriever import LanceRetriever
from src.rag.generator import GroqGenerator
from src.rag.guardrails import GuardrailEngine
from src.analytics.latency_evaluator import LatencyBenchmarkEvaluator

TEST_QUERIES = [
    "What is the capital of India?", "When did Goa join India?", "What is the capital of UP?",
    "What is the capital of Madhya Pradesh?", "What is the longest river in India?",
    "What is the capital of Maharashtra?", "What is the capital of Rajasthan?",
    "What is the capital of Karnataka?", "What is the capital of Tamil Nadu?",
    "What is the capital of Gujarat?", "What is the capital of Bihar?",
    "What is the capital of West Bengal?", "What is the capital of Punjab?",
    "What is the capital of Kerala?", "What is the capital of Telangana?",
    "What is the capital of Odisha?", "What is the capital of Assam?",
    "Who wrote national anthem of India?", "When did India get independence?",
    "Who was first prime minister of India?", "What is MS MARCO dataset?",
    "What is the highest peak in India?", "What is the largest desert in India?",
    "What is the capital city of Goa?", "Where is Panaji located?",
    "What river flows through Lucknow?", "What is Silicon Valley of India?",
    "What is Pink City of India?", "What is Temple City of Odisha?",
    "Where does Ganga river originate?", "Which glacier does Ganga originate from?",
    "When was Goa liberated?", "What was Operation Vijay?",
    "Who composed Jana Gana Mana?", "What is the capital of Andhra Pradesh?",
    "Tell me about the Thar Desert in India", "What is the financial capital of India?",
    "Which sea is near Tamil Nadu?", "What is the official capital of India?",
    "Who was the prime minister in 1947?", "What is the state capital of Assam?",
    "What is the capital of Haryana?", "What is the capital of Gandhinagar?",
    "What is the capital of Trivandrum?", "What is the capital of Amaravati?",
    "Where is Kanchenjunga located?", "Which Indian state has Bhopal as capital?",
    "Which Indian state has Jaipur as capital?", "Which state has Mumbai as capital?",
    "What is the capital of India in New Delhi?"
]

async def main():
    retriever = LanceRetriever()
    generator = GroqGenerator()
    guardrails = GuardrailEngine()
    
    evaluator = LatencyBenchmarkEvaluator(retriever, generator, guardrails)
    await evaluator.benchmark_pipeline(TEST_QUERIES, warmup_count=5)

if __name__ == "__main__":
    asyncio.run(main())