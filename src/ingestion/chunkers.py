import re
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class MultiStrategyChunker:
    """
    Implements multi-strategy chunking:
    1. Fixed-size overlapping chunks (Character-based)
    2. Semantic / Sentence-boundary splitting with metadata awareness
    3. Hierarchical / Document-level chunking
    """
    def __init__(self, fixed_size: int = 350, fixed_overlap: int = 50):
        self.fixed_splitter = RecursiveCharacterTextSplitter(
            chunk_size=fixed_size,
            chunk_overlap=fixed_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _sentence_split(self, text: str) -> List[str]:
        # Splits cleanly on sentence endings across English and Indic scripts
        sentences = re.split(r'(?<=[.?!।])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 15]

    def process_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        doc_id = doc["doc_id"]
        lang = doc.get("language", "en")
        
        for p_idx, passage in enumerate(doc["passages"]):
            # Strategy A: Fixed-Size with Overlap
            fixed_parts = self.fixed_splitter.split_text(passage)
            for f_idx, part in enumerate(fixed_parts):
                chunks.append({
                    "id": f"{doc_id}_p{p_idx}_f{f_idx}",
                    "text": part,
                    "doc_id": doc_id,
                    "strategy": "fixed_overlap",
                    "language": lang,
                    "char_count": len(part)
                })

            # Strategy B: Sentence / Semantic Boundary Chunking
            sentences = self._sentence_split(passage)
            current_chunk = []
            current_len = 0
            s_idx = 0
            
            for sentence in sentences:
                current_chunk.append(sentence)
                current_len += len(sentence)
                if current_len >= 250:
                    merged = " ".join(current_chunk)
                    chunks.append({
                        "id": f"{doc_id}_p{p_idx}_s{s_idx}",
                        "text": merged,
                        "doc_id": doc_id,
                        "strategy": "sentence_boundary",
                        "language": lang,
                        "char_count": len(merged)
                    })
                    current_chunk = []
                    current_len = 0
                    s_idx += 1
                    
            if current_chunk:
                merged = " ".join(current_chunk)
                chunks.append({
                    "id": f"{doc_id}_p{p_idx}_s{s_idx}",
                    "text": merged,
                    "doc_id": doc_id,
                    "strategy": "sentence_boundary",
                    "language": lang,
                    "char_count": len(merged)
                })

        return chunks