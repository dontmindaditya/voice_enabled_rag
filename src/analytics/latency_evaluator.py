import time
import asyncio
import numpy as np
from typing import List, Dict, Any

class LatencyEvaluator:
    """
    Computes P50, P70, P100 latency analytics across a test suite of queries.
    Satisfies Competition Requirement #4.
    """
    def __init__(self, retriever, generator, guardrails):
        self.retriever = retriever
        self.generator = generator
        self.guardrails = guardrails

    async def benchmark_pipeline(self, test_queries: List[str]) -> Dict[str, Any]:
        print(f"[*] Warming up connection & model cache...")
        # Warm-up request (not included in metrics)
        dummy_ret = self.retriever.retrieve("warmup")
        await self.generator.generate_grounded_answer("warmup", dummy_ret["context"])
        
        print(f"[*] Starting latency benchmarking across {len(test_queries)} queries...")
        latencies = []
        retrieval_times = []
        generation_times = []
        results_log = []

        for query in test_queries:
            t_start = time.perf_counter()
            
            # 1. Input Guardrail (<1ms)
            is_safe, _ = self.guardrails.validate_input(query)
            if not is_safe:
                continue

            # 2. Vector Retrieval (~15ms)
            ret_res = self.retriever.retrieve(query)
            ret_time = ret_res["latency_ms"]

            # 3. LLM Generation (<120ms)
            gen_res = await self.generator.generate_grounded_answer(query, ret_res["context"])
            gen_time = gen_res["latency_ms"]

            # 4. Output Guardrail (<1ms)
            self.guardrails.validate_groundedness(gen_res["answer"], ret_res["context"])
            
            total_time = (time.perf_counter() - t_start) * 1000
            
            latencies.append(total_time)
            retrieval_times.append(ret_time)
            generation_times.append(gen_time)

            results_log.append({
                "query": query,
                "total_ms": round(total_time, 2),
                "ret_ms": ret_time,
                "gen_ms": gen_time
            })
            
            # 2.1s pause to avoid hitting free-tier 30 RPM rate limits
            await asyncio.sleep(2.1)

        # Compute Percentiles
        p50 = float(np.percentile(latencies, 50))
        p70 = float(np.percentile(latencies, 70))
        p100 = float(np.percentile(latencies, 100))

        summary = {
            "total_queries_tested": len(latencies),
            "p50_latency_ms": round(p50, 2),
            "p70_latency_ms": round(p70, 2),
            "p100_latency_ms": round(p100, 2),
            "avg_retrieval_ms": round(float(np.mean(retrieval_times)), 2),
            "avg_generation_ms": round(float(np.mean(generation_times)), 2),
            "logs": results_log
        }

        print("\n" + "=" * 45)
        print("  LATENCY ANALYTICS BENCHMARK REPORT")
        print("=" * 45)
        print(f"Total Queries : {summary['total_queries_tested']}")
        print(f"P50 Latency   : {summary['p50_latency_ms']} ms")
        print(f"P70 Latency   : {summary['p70_latency_ms']} ms")
        print(f"P100 Latency  : {summary['p100_latency_ms']} ms")
        print("=" * 45 + "\n")

        return summary