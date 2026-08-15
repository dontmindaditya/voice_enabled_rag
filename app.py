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

# Custom black-and-white theme styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #09090b !important;
    color: #f4f4f5 !important;
}

[data-testid="stHeader"] {
    background-color: rgba(9, 9, 11, 0.8) !important;
    backdrop-filter: blur(8px) !important;
}

h1, h2, h3, h4, h5, h6, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2 {
    font-family: 'Outfit', sans-serif !important;
    color: #ffffff !important;
}

/* Custom CSS to style all buttons as black and white */
button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #ffffff !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 500 !important;
    font-family: 'Outfit', sans-serif !important;
    transition: all 0.2s ease-in-out !important;
}

button:hover, [data-testid="stBaseButton-secondary"]:hover, [data-testid="stBaseButton-primary"]:hover {
    background-color: #000000 !important;
    color: #ffffff !important;
    border-color: #ffffff !important;
    box-shadow: 0 0 10px rgba(255, 255, 255, 0.15) !important;
}

/* Secondary Button styling overrides */
div[data-testid="stButton"] button[type="secondary"] {
    background-color: transparent !important;
    color: #ffffff !important;
    border: 1px solid #27272a !important;
}

div[data-testid="stButton"] button[type="secondary"]:hover {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-color: #ffffff !important;
}

/* Input Fields styling */
input, textarea, [data-testid="stTextInput"] input {
    background-color: #09090b !important;
    color: #ffffff !important;
    border: 1px solid #27272a !important;
    border-radius: 6px !important;
}

/* File Uploader styling */
[data-testid="stFileUploader"] {
    background-color: #121214 !important;
    border: 1px dashed #27272a !important;
    border-radius: 8px !important;
    padding: 10px !important;
}

/* Metrics Cards styling */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700 !important;
}

[data-testid="stMetricDelta"] {
    color: #a1a1aa !important;
}

/* Custom styled Alert/Success Boxes */
.custom-alert {
    background-color: #09090b;
    border: 1px solid #ffffff;
    border-left: 5px solid #ffffff;
    border-radius: 6px;
    padding: 1.2rem;
    margin-bottom: 1.5rem;
}

/* Hide default streamlit decoration */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# SVG Icons
mic_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>"""

db_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/></svg>"""

check_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><polyline points="20 6 9 17 4 12"/></svg>"""

shield_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>"""

input_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><polygon points="5 3 19 12 5 21 5 3"/></svg>"""

response_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>"""

bubble_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>"""

timer_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>"""

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

# Run ID for recording again reset trick
if "run_id" not in st.session_state:
    st.session_state.run_id = 0

# Page Header
st.markdown(f"""
<div style="display: flex; align-items: center; margin-top: 1rem; margin-bottom: 0.2rem;">
    {mic_svg}
    <h1 style="margin: 0; margin-left: 12px; font-size: 2.2rem; font-weight: 700;">Low-Latency Voice RAG</h1>
</div>
""", unsafe_allow_html=True)
st.markdown('<p style="color: #a1a1aa; font-size: 0.95rem; margin-top: 0rem; margin-bottom: 1.5rem;">HH Goa 2026 Shortlisting Task 2 — Target Latency: &lt; 200ms</p>', unsafe_allow_html=True)

# Knowledge Base Information Card
st.markdown(f"""
<div style="background-color: #121214; border: 1px solid #27272a; border-radius: 8px; padding: 1.2rem; margin-bottom: 2rem;">
    <div style="display: flex; align-items: flex-start; margin-bottom: 0.8rem;">
        <div style="margin-top: 2px;">{db_svg}</div>
        <div style="margin-left: 10px;">
            <strong>Knowledge Base:</strong> This RAG pipeline is indexed on the <strong>AI4Bharat MSMARCO-XI</strong> Indic Benchmark dataset.
        </div>
    </div>
    <div style="display: flex; align-items: flex-start; margin-bottom: 0.8rem;">
        <div style="margin-top: 2px;">{check_svg}</div>
        <div style="margin-left: 10px;">
            To test <strong>grounded answers</strong>, ask questions related to India (geography, history, capitals, states).
        </div>
    </div>
    <div style="display: flex; align-items: flex-start;">
        <div style="margin-top: 2px;">{shield_svg}</div>
        <div style="margin-left: 10px;">
            For un-indexed queries, the <strong>Safety Guardrail</strong> will strictly refuse to hallucinate.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(f"""
    <h3 style="margin-top: 0.5rem; margin-bottom: 1.2rem; display: flex; align-items: center; font-size: 1.3rem;">
        {input_svg} <span style="margin-left: 10px;">1. Audio Input &amp; Query</span>
    </h3>
    """, unsafe_allow_html=True)
    
    # Dynamic keys to clear widget state on reset
    audio_input = st.audio_input("Record your question", key=f"audio_{st.session_state.run_id}")
    uploaded_file = st.file_uploader("Or upload an audio file (.wav, .mp3)", type=["wav", "mp3"], key=f"file_{st.session_state.run_id}")
    text_fallback = st.text_input("Or enter text query directly (Benchmark Mode):", key=f"text_{st.session_state.run_id}")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        submit_btn = st.button("Run RAG Pipeline", use_container_width=True)
    with col_btn2:
        record_again_btn = st.button("Record Again", use_container_width=True)
        if record_again_btn:
            st.session_state.run_id += 1
            st.rerun()

with col2:
    st.markdown(f"""
    <h3 style="margin-top: 0.5rem; margin-bottom: 1.2rem; display: flex; align-items: center; font-size: 1.3rem;">
        {response_svg} <span style="margin-left: 10px;">2. Response &amp; Groundedness</span>
    </h3>
    """, unsafe_allow_html=True)
    
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
            st.error(f"Input Blocked by Guardrail: {msg}")
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
            st.markdown(f'<div style="font-size: 1.1rem; margin-bottom: 0.8rem;">{bubble_svg} <b>Transcribed Query:</b> <code>{query_text}</code></div>', unsafe_allow_html=True)
            st.markdown("### Answer")
            
            # Custom styled black and white response card
            st.markdown(f"""
            <div class="custom-alert">
                <p style="margin: 0; font-size: 1.1rem; color: #ffffff; line-height: 1.6; font-weight: 400;">{raw_answer}</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Retrieved Context & Grounding Audit"):
                st.markdown(f"**Grounding Status:** `{ground_check['status']}` (Overlap: {ground_check.get('overlap_ratio', 1.0) * 100:.0f}%)")
                st.text(context)

        with metrics_container.container():
            st.markdown(f"### {timer_svg} <span style='margin-left: 10px;'>Execution Latency Breakdown</span>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("STT (Sarvam)", f"{stt_latency:.1f} ms")
            c2.metric("Retrieval (LanceDB)", f"{ret_latency:.1f} ms")
            c3.metric("LLM (Groq LPU)", f"{gen_latency:.1f} ms")
            c4.metric("Total Latency", f"{total_latency:.1f} ms", delta=f"{200 - total_latency:.1f} ms under target" if total_latency < 200 else "-Exceeded")

            # Visual Timing Bar styled black and white
            latency_df = pd.DataFrame({
                "Stage": ["STT", "Retrieval", "LLM Generation"],
                "Latency (ms)": [stt_latency, ret_latency, gen_latency]
            })
            st.bar_chart(latency_df.set_index("Stage"), color="#ffffff")