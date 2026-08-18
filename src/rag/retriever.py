import time
import lancedb
from typing import List, Dict, Any
from fastembed import TextEmbedding

class LanceRetriever:
    """
    Sub-10ms in-process vector retrieval from local LanceDB table.
    """
    def __init__(self, db_path: str = "./vector_db/msmarco.lancedb", table_name: str = "msmarco_chunks"):
        self.db = lancedb.connect(db_path)
        try:
            self.table = self.db.open_table(table_name)
        except Exception as e:
            print(f"[*] Table '{table_name}' not found or corrupt. Rebuilding index dynamically...")
            from src.ingestion.load_dataset import fetch_msmarco_passages
            from src.ingestion.chunkers import MultiStrategyChunker
            from src.ingestion.indexer import VectorIndexer
            
            passages = fetch_msmarco_passages(limit=300)
            chunker = MultiStrategyChunker()
            chunks = []
            for passage in passages:
                chunks.extend(chunker.process_document(passage))
                
            indexer = VectorIndexer(db_path=db_path)
            self.table = indexer.build_index(chunks, table_name=table_name)
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=1)

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        t0 = time.perf_counter()
        
        # 1. Embed query on local CPU (~3-5ms)
        query_vector = list(self.embed_model.embed([query]))[0]
        
        # 2. In-process vector similarity search (<5ms)
        results = self.table.search(query_vector).limit(top_k).to_list()
        
        # Check if the top result is a strong match (distance <= 0.65, i.e., similarity >= 0.35)
        is_relevant = False
        if results:
            top_distance = results[0].get("_distance", 1.0)
            if top_distance <= 0.65:
                is_relevant = True
                
        latency = (time.perf_counter() - t0) * 1000
        
        if is_relevant:
            context_chunks = [r["text"] for r in results]
            combined_context = "\n---\n".join(context_chunks)
        else:
            combined_context = ""
            
        return {
            "context": combined_context,
            "raw_results": results,
            "latency_ms": round(latency, 2)
        }