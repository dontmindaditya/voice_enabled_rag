import os
from src.ingestion.load_dataset import fetch_msmarco_passages
from src.ingestion.chunkers import MultiStrategyChunker
from src.ingestion.indexer import VectorIndexer

def main():
    print("=== Rebuilding LanceDB Index with Comprehensive Indian Knowledge Base ===")
    
    # 1. Fetch passages
    passages = fetch_msmarco_passages(limit=300)
    
    # 2. Chunk passages
    chunker = MultiStrategyChunker()
    chunks = []
    for passage in passages:
        chunks.extend(chunker.process_document(passage))
        
    print(f"[*] Processed {len(passages)} passages into {len(chunks)} chunks.")
    
    # 3. Index chunks
    db_path = "./vector_db/msmarco.lancedb"
    table_name = "msmarco_chunks"
    
    indexer = VectorIndexer(db_path=db_path)
    indexer.build_index(chunks, table_name=table_name)
    print("[OK] LanceDB Index successfully rebuilt!")

if __name__ == "__main__":
    main()
