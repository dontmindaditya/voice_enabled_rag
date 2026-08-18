import os
import time
import asyncio
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Import optimized RAG modules
from src.stt.sarvam_transcribe import SarvamSTT
from src.rag.retriever import LanceRetriever
from src.rag.generator import GroqGenerator
from src.rag.guardrails import GuardrailEngine

# Streamlit Page Config
st.set_page_config(
    page_title="Voice-Enabled RAG System",
    page_icon="🎙️",
    layout="wide"
)

# Custom premium black-and-white theme styling
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

/* Custom styled Streamlit buttons */
div[data-testid="stButton"] button {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #ffffff !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 500 !important;
    font-family: 'Outfit', sans-serif !important;
    transition: all 0.2s ease-in-out !important;
}

div[data-testid="stButton"] button:hover {
    background-color: #000000 !important;
    color: #ffffff !important;
    border-color: #ffffff !important;
    box-shadow: 0 0 10px rgba(255, 255, 255, 0.15) !important;
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
timer_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>"""

# Header & Info Banner
st.markdown(f"""
<div style="display: flex; align-items: center; margin-top: 1rem; margin-bottom: 0.2rem;">
    {mic_svg}
    <h1 style="margin: 0; margin-left: 12px; font-size: 2.2rem; font-weight: 700;">Low-Latency Voice-Enabled RAG System</h1>
</div>
""", unsafe_allow_html=True)
st.caption("HH Goa 2026 Shortlisting Task 2 — Target Latency: < 200ms")

st.info(
    "ℹ️ **Knowledge Base Scope:** Indexed on the **AI4Bharat MSMARCO-XI** Indic Benchmark dataset. "
    "Answers factual queries on Indian geography, history, and state capitals. "
    "Unrelated queries will trigger the **Groundedness Guardrail** to prevent hallucinations."
)

@st.cache_resource
def load_pipeline():
    retriever = LanceRetriever()
    generator = GroqGenerator()
    guardrails = GuardrailEngine()
    stt = SarvamSTT()
    
    # Warm up the Groq connection pool synchronously during resource load
    # so that the user's very first query is sub-200ms
    try:
        generator._generate_grounded_answer_sync("warmup", "warmup")
    except Exception:
        pass
        
    return retriever, generator, guardrails, stt

retriever, generator, guardrails, stt = load_pipeline()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Audio Input & Query")
    audio_input = st.audio_input("Record your question:")
    uploaded_file = st.file_uploader("Or upload an audio file (.wav, .mp3)", type=["wav", "mp3"])
    text_query = st.text_input("Or enter query directly (Benchmark Mode):", placeholder="e.g., What is the capital of Madhya Pradesh?")
    submit_btn = st.button("🚀 Run RAG Pipeline", use_container_width=True)

with col2:
    st.subheader("2. Response & Groundedness")
    
    if submit_btn or audio_input or uploaded_file or text_query:
        query_text = ""
        stt_latency = 0.0
        
        # --- Stage 1: Speech-to-Text ---
        if audio_input is not None:
            with open("temp_record.wav", "wb") as f:
                f.write(audio_input.getvalue())
            stt_res = stt.transcribe("temp_record.wav")
            query_text = stt_res.get("transcript", "")
            stt_latency = stt_res.get("latency_ms", 0.0)
            if os.path.exists("temp_record.wav"):
                os.remove("temp_record.wav")
        elif uploaded_file is not None:
            with open("temp_upload.wav", "wb") as f:
                f.write(uploaded_file.getvalue())
            stt_res = stt.transcribe("temp_upload.wav")
            query_text = stt_res.get("transcript", "")
            stt_latency = stt_res.get("latency_ms", 0.0)
            if os.path.exists("temp_upload.wav"):
                os.remove("temp_upload.wav")
        elif text_query.strip():
            query_text = text_query.strip()
            stt_latency = 0.0

        if query_text:
            st.markdown(f"🗣️ **Transcribed Query:** `{query_text}`")
            
            # --- Stage 2: Input Guardrail ---
            is_safe, reason = guardrails.validate_input(query_text)
            if not is_safe:
                st.error(f"🛑 Security Guardrail Blocked: {reason}")
            else:
                # --- Stage 3: Retrieval ---
                ret_res = retriever.retrieve(query_text)
                ret_latency = ret_res["latency_ms"]
                context = ret_res["context"]
                
                # --- Stage 4: Generation ---
                t_gen_start = time.perf_counter()
                gen_res = asyncio.run(generator.generate_grounded_answer(query_text, context))
                gen_latency = gen_res["latency_ms"]
                raw_answer = gen_res["answer"]
                
                # --- Stage 5: Output Grounding Check ---
                # Fix unpacking crash: validate_groundedness returns a dict
                ground_res = guardrails.validate_groundedness(raw_answer, context)
                is_grounded = ground_res["grounded"]
                ground_status = ground_res["status"]
                
                core_rag_latency = ret_latency + gen_latency
                total_latency = stt_latency + core_rag_latency

                # Render Answer
                st.subheader("Answer")
                st.markdown(f"""
                <div class="custom-alert">
                    <p style="margin: 0; font-size: 1.1rem; color: #ffffff; line-height: 1.6; font-weight: 400;">{raw_answer}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Retrieved Context Dropdown
                with st.expander("🔍 Retrieved Context & Grounding Audit"):
                    st.write(f"**Groundedness Status:** `{ground_status}`")
                    st.write(f"**Context Used:**\n{context}")

                # Real-Time Latency Breakdown
                st.subheader("⚡ Execution Latency Breakdown")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("STT (Sarvam)", f"{stt_latency:.1f} ms")
                m2.metric("Retrieval (LanceDB)", f"{ret_latency:.1f} ms")
                m3.metric("LLM (Groq LPU)", f"{gen_latency:.1f} ms")
                
                # Target evaluation badge
                delta_status = "Sub-200ms Target Met ✅" if core_rag_latency < 200 else "Exceeded 200ms ❌"
                m4.metric("Core RAG Total", f"{core_rag_latency:.1f} ms", delta=delta_status)

                # Visual Timing Bar styled black and white
                latency_df = pd.DataFrame({
                    "Stage": ["STT", "Retrieval", "LLM Generation"],
                    "Latency (ms)": [stt_latency, ret_latency, gen_latency]
                })
                st.bar_chart(latency_df.set_index("Stage"), color="#ffffff")