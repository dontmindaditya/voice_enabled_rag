import os
import lancedb
from typing import List, Dict, Any
from fastembed import TextEmbedding

class VectorIndexer:
    """
    Zero-cost In-Process Indexer utilizing local CPU ONNX embeddings
    and an embedded LanceDB database (<10ms lookup speed).
    """
    def __init__(self, db_path: str = "./vector_db/msmarco.lancedb"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = lancedb.connect(db_path)
        print("[*] Initializing local FastEmbed model (BAAI/bge-small-en-v1.5)...")
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def build_index(self, chunks: List[Dict[str, Any]], table_name: str = "msmarco_chunks"):
        print(f"[*] Generating local embeddings for {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]
        
        # FastEmbed uses local ONNX runtime (~5ms/batch on CPU)
        embeddings = list(self.embed_model.embed(texts))
        
        table_data = []
        for chunk, emb in zip(chunks, embeddings):
            chunk["vector"] = emb.tolist()
            table_data.append(chunk)

        print(f"[*] Creating LanceDB table '{table_name}'...")
        table = self.db.create_table(table_name, data=table_data, mode="overwrite")
        try:
            # LanceDB IVF-PQ index requires at least 256 rows to train
            if len(table_data) >= 256:
                table.create_index(metric="cosine")
                print(f"[OK] Index successfully written with {len(table_data)} vectors!")
            else:
                print(f"[OK] Table successfully written with {len(table_data)} vectors (flat index used for small dataset)!")
        except Exception as e:
            print(f"[!] Warning: Could not create IVF-PQ index ({e}). Falling back to brute-force flat search.")
        return table