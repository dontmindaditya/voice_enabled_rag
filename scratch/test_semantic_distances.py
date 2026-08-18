import asyncio
from src.rag.retriever import LanceRetriever

def main():
    retriever = LanceRetriever()
    
    test_queries = [
        "who is 1st pm",
        "gandhi jayanti date",
        "what is the national language of india",
        "How many states are there in India?",
        "what is the capital of india",
        "who was the first prime minister",
        "tell me about red fort",
        "gandhi jayanti date of birth",
        "how to build an airplane"
    ]
    
    print("--- Testing Semantic Distance Matcher ---")
    for q in test_queries:
        res = retriever.retrieve(q, top_k=1)
        raw = res["raw_results"][0]
        # LanceDB distance is 1 - cosine similarity
        similarity = 1.0 - raw["_distance"]
        print(f"Query: '{q}'")
        print(f"  Top Match: '{raw['text'][:80]}...'")
        print(f"  Similarity: {similarity:.4f} (Distance: {raw['_distance']:.4f})")

if __name__ == "__main__":
    main()
