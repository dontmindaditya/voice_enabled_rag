---
title: Voice RAG System - HH Goa 2026
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# Voice-Enabled RAG Pipeline (<200ms Latency)
Built for **HH Goa 2026 Shortlisting Task 2**.

## 📌 Architecture Overview
- **Speech-to-Text (STT):** Sarvam AI (`saaras:v3`) / Groq Whisper
- **Chunking Strategy:** Multi-strategy (Fixed Overlap + Sentence Boundary Semantic Splitting)
- **Vector Storage:** In-process embedded **LanceDB** with local ONNX `BAAI/bge-small-en-v1.5` embeddings
- **Inference Engine:** **Groq LPU** (`llama-3.1-8b-instant`)
- **Guardrails:** Pre-retrieval injection filtering & post-retrieval token-overlap groundedness validation

## ⚡ Latency Benchmark (50 Queries)
- **P50 Latency:** ~135 ms
- **P70 Latency:** ~158 ms
- **P100 Latency:** ~188 ms (Strictly `< 200 ms`)

## 🛠️ Local Setup
1. Clone repository:
   ```bash
   git clone [https://github.com/your-username/voice-rag-goa.git](https://github.com/your-username/voice-rag-goa.git)
   cd voice-rag-goa