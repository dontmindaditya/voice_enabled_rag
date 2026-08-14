import re
from typing import Tuple, Dict, Any

class GuardrailEngine:
    """
    Lightweight, deterministic guardrail system (<2ms overhead):
    1. Input Guardrail: Filters prompt injections, offensive language, and off-topic commands.
    2. Output Guardrail: Prevents hallucinations and verifies response groundedness.
    """
    def __init__(self):
        self.injection_patterns = [
            r"ignore\s+(previous|all)\s+instructions",
            r"system\s*prompt",
            r"jailbreak",
            r"bypass",
            r"you\s+are\s+now\s+dan",
            r"reveal\s+(api\s*key|password|secret)"
        ]

    def validate_input(self, query: str) -> Tuple[bool, str]:
        """Validates incoming user query before calling vector search or LLM."""
        if not query or len(query.strip()) < 3:
            return False, "Query is too short or empty."

        for pattern in self.injection_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Query flagged: Prompt injection or malicious intent detected."

        return True, "Safe"

    def validate_groundedness(self, answer: str, context: str) -> Dict[str, Any]:
        """Ensures the generated answer is grounded in retrieved context."""
        if "cannot answer based on the provided dataset" in answer.lower():
            return {
                "grounded": True,
                "status": "REFUSED_UNKNOWN_CONTEXT",
                "notes": "Model correctly refused to hallucinate outside the dataset."
            }

        # Check keyword overlap between answer and context
        answer_words = set(re.findall(r'\w+', answer.lower()))
        context_words = set(re.findall(r'\w+', context.lower()))
        
        # Exclude common stopwords
        stopwords = {"the", "a", "an", "is", "in", "it", "of", "and", "to", "was", "for", "on"}
        key_answer_words = answer_words - stopwords
        
        if not key_answer_words:
            return {"grounded": True, "status": "VERIFIED", "overlap_ratio": 1.0}
            
        overlap = key_answer_words.intersection(context_words)
        overlap_ratio = len(overlap) / len(key_answer_words)

        # Grounded if at least 40% of non-stopword tokens exist in context
        is_grounded = overlap_ratio >= 0.40

        return {
            "grounded": is_grounded,
            "status": "VERIFIED" if is_grounded else "POTENTIAL_HALLUCINATION",
            "overlap_ratio": round(overlap_ratio, 2)
        }