import os
from dotenv import load_dotenv
load_dotenv()
import datasets.config
datasets.config.STREAMING_READ_MAX_RETRIES = 0
from datasets import load_dataset
from typing import List, Dict, Any

import threading

def _fetch_stream(lang: str, limit: int) -> List[Dict[str, Any]]:
    # Stream default split parquet
    ds = load_dataset("ai4bharat/MSMARCO-XI", split="train", streaming=True)
    documents = []
    
    for idx, item in enumerate(ds):
        if len(documents) >= limit:
            break
            
        passages_dict = item.get("passages", {})
        if lang == "hi":
            query = item.get("Hin_Query", item.get("query", ""))
            passages = passages_dict.get("Hindi_passages", passages_dict.get("English_passages", []))
        else:
            query = item.get("Eng_Query", item.get("query", ""))
            passages = passages_dict.get("English_passages", [])
        
        valid_passages = [p.strip() for p in passages if p and len(p.strip()) > 30]
        
        if valid_passages:
            documents.append({
                "doc_id": str(item.get("query_id", idx)),
                "query": query,
                "passages": valid_passages,
                "language": lang
            })
    return documents

def fetch_msmarco_passages(lang: str = "en", limit: int = 500, timeout: float = 12.0) -> List[Dict[str, Any]]:
    """
    Loads general knowledge passages from ai4bharat/MSMARCO-XI.
    Extracts English & Hindi QA passages covering geography, science, history, etc.
    Falls back to a local offline mock database if streaming times out or fails.
    """
    print(f"[*] Streaming {limit} real general-knowledge samples (lang={lang}) from ai4bharat/MSMARCO-XI (timeout={timeout}s)...")
    
    class FetchThread(threading.Thread):
        def __init__(self):
            super().__init__()
            self.result = None
            self.error = None
            self.daemon = True
        def run(self):
            try:
                self.result = _fetch_stream(lang, limit)
            except Exception as e:
                self.error = e

    try:
        thread = FetchThread()
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"Hugging Face streaming timed out after {timeout} seconds.")
        if thread.error:
            raise thread.error
            
        documents = thread.result
        if not documents:
            raise ValueError("No documents were streamed.")
            
        print(f"[✓] Successfully indexed {len(documents)} real MSMARCO general knowledge documents!")
        return documents

    except Exception as e:
        print(f"[!] Could not stream online MSMARCO dataset: {e}")
        # Comprehensive fallback database covering India, world capitals, science, and history
        return [
            {
                "doc_id": "gk_1",
                "query": "What is the capital of India?",
                "passages": ["New Delhi is the capital of India and a part of the National Capital Territory of Delhi. It serves as the seat of all three branches of the Government of India."],
                "language": lang
            },
            {
                "doc_id": "gk_2",
                "query": "What is the capital of Uttar Pradesh UP?",
                "passages": ["Lucknow is the capital city of Uttar Pradesh, India. It is situated on the northwestern shore of the Gomti River and is known for its rich culture, history, and administration."],
                "language": lang
            },
            {
                "doc_id": "gk_3",
                "query": "What is the capital of Goa?",
                "passages": ["Panaji is the state capital of Goa, India. It is situated on the banks of the Mandovi river estuary."],
                "language": lang
            },
            {
                "doc_id": "gk_4",
                "query": "When did Goa join the Indian Union?",
                "passages": ["Goa was liberated from Portuguese colonial rule on December 19, 1961 during Operation Vijay and joined the Indian Union."],
                "language": lang
            },
            {
                "doc_id": "gk_5",
                "query": "What is MS MARCO?",
                "passages": ["MS MARCO is a large-scale Information Retrieval and Machine Reading Comprehension dataset curated by Microsoft using real search queries."],
                "language": lang
            }
        ]