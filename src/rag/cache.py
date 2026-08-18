import time
from typing import Optional, Dict, Any

class SimpleSemanticCache:
    """
    Sub-millisecond in-memory cache for repeated & high-frequency queries.
    """
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds

    def _normalize(self, query: str) -> str:
        return "".join(e for e in query.lower().strip() if e.isalnum() or e.isspace())

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        key = self._normalize(query)
        entry = self.cache.get(key)
        if entry:
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["data"]
            del self.cache[key]
        return None

    def set(self, query: str, data: Dict[str, Any]):
        key = self._normalize(query)
        self.cache[key] = {
            "timestamp": time.time(),
            "data": data
        }
