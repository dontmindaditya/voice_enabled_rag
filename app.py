import os
import lancedb
import time
import asyncio
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Import internal modules
from src.stt.sarvam_transcribe import SarvamTranscriber
from src.rag.retriever import LanceRetriever
from src.rag.guardrails import GuardrailEngine
from src.rag.generator import GroqGenerator
from src.ingestion.load_dataset import fetch_msmarco_passages
from src.ingestion.chunkers import MultiStrategyChunker
from src.ingestion.indexer import VectorIndexer

# Streamlit Page Config
st.set_page_config(
    page_title="Voice-Enabled RAG | HH Goa 2026",
    page_icon="🎙️",
    layout="wide"
)

@st.cache_resource
def initialize_system():
    """Initializes and caches embedding models, vector index, and harness."""
    db_path = "./vector_db/msmarco.lancedb"
    table_name = "msmarco_chunks"
    
    # Auto-build database if not present or table doesn't exist
    db_exists_and_has_table = False
    try:
        db = lancedb.connect(db_path)
        db.open_table(table_name)
        db_exists_and_has_table = True
    except Exception:
        pass

    if not db_exists_and_has_table:
        with st.spinner("Initializing Vector Index from MSMARCO-XI dataset..."):
            docs = fetch_msmarco_passages(lang="en", limit=100)
            chunker = MultiStrategyChunker()
            all_chunks = []
            for doc in docs:
                all_chunks.extend(chunker.process_document(doc))
            indexer = VectorIndexer(db_path=db_path)
            indexer.build_index(all_chunks, table_name=table_name)
            
    transcriber = SarvamTranscriber()
    retriever = LanceRetriever(db_path=db_path, table_name=table_name)
    guardrails = GuardrailEngine()
    generator = GroqGenerator()
    
    return transcriber, retriever, guardrails, generator

transcriber, retriever, guardrails, generator = initialize_system()

# UI Layout
st.title("🎙️ Low-Latency Voice-Enabled RAG System")
st.caption("HH Goa 2026 Shortlisting Task 2 — Target Latency: < 200ms")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Audio Input & Query")
    audio_input = st.audio_input("Record your question")
    uploaded_file = st.file_uploader("Or upload an audio file (.wav, .mp3)", type=["wav", "mp3"])
    text_fallback = st.text_input("Or enter text query directly (Benchmark Mode):")
    
    submit_btn = st.button("🚀 Run RAG Pipeline", use_container_width=True)

with col2:
    st.subheader("2. Response & Groundedness")
    output_container = st.empty()
    metrics_container = st.empty()

if submit_btn:
    with st.spinner("Executing pipeline..."):
        t_pipeline_start = time.perf_counter()
        query_text = ""
        stt_latency = 0.0
        
        # Phase 1: STT
        if audio_input or uploaded_file:
            audio_source = audio_input if audio_input else uploaded_file
            temp_path = f"temp_audio_{int(time.time())}.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_source.read())
            
            stt_result = transcriber.transcribe(temp_path)
            query_text = stt_result["transcript"]
            stt_latency = stt_result["latency_ms"]
            if os.path.exists(temp_path):
                os.remove(temp_path)
        elif text_fallback:
            query_text = text_fallback
            stt_latency = 0.0
        else:
            st.warning("Please record audio, upload a file, or enter a text query.")
            st.stop()

        # Phase 2: Input Guardrail
        is_safe, msg = guardrails.validate_input(query_text)
        if not is_safe:
            st.error(f"❌ Input Blocked by Guardrail: {msg}")
            st.stop()

        # Phase 3: LanceDB Vector Retrieval
        ret_result = retriever.retrieve(query_text, top_k=3)
        ret_latency = ret_result["latency_ms"]
        context = ret_result["context"]

        # Phase 4: Grounded Generation (Groq Llama 3.1)
        gen_result = asyncio.run(generator.generate_grounded_answer(query_text, context))
        gen_latency = gen_result["latency_ms"]
        raw_answer = gen_result["answer"]

        # Phase 5: Output Guardrail (Groundedness Check)
        ground_check = guardrails.validate_groundedness(raw_answer, context)
        t_pipeline_end = time.perf_counter()
        total_latency = (t_pipeline_end - t_pipeline_start) * 1000

        # Display Outputs
        with output_container.container():
            st.markdown(f"**🗣️ Transcribed Query:** `{query_text}`")
            st.markdown("### Answer")
            st.success(raw_answer)
            
            with st.expander("🔍 Retrieved Context & Grounding Audit"):
                st.markdown(f"**Grounding Status:** `{ground_check['status']}` (Overlap: {ground_check.get('overlap_ratio', 1.0) * 100:.0f}%)")
                st.text(context)

        with metrics_container.container():
            st.markdown("### ⚡ Execution Latency Breakdown")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("STT (Sarvam)", f"{stt_latency:.1f} ms")
            c2.metric("Retrieval (LanceDB)", f"{ret_latency:.1f} ms")
            c3.metric("LLM (Groq LPU)", f"{gen_latency:.1f} ms")
            c4.metric("Total Latency", f"{total_latency:.1f} ms", delta=f"{200 - total_latency:.1f} ms under target" if total_latency < 200 else "-Exceeded")

            # Visual Timing Bar
            latency_df = pd.DataFrame({
                "Stage": ["STT", "Retrieval", "LLM Generation"],
                "Latency (ms)": [stt_latency, ret_latency, gen_latency]
            })
            st.bar_chart(latency_df.set_index("Stage"))