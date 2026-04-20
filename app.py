"""
Faith-Aware Wellness Companion — Streamlit UI

Run:  streamlit run app.py
"""
from __future__ import annotations

import tempfile

import streamlit as st

try:
    from audio_recorder_streamlit import audio_recorder
except ImportError:
    audio_recorder = None

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Faith-Aware Wellness Companion",
    page_icon="🌙",
    layout="centered",
)

# ── Lazy-load heavy components once per session ─────────────────────────────
@st.cache_resource(show_spinner="Loading retrieval index …")
def get_retriever():
    from src.retriever import Retriever
    return Retriever()


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    top_k = st.slider("Passages to retrieve (top-k)", 1, 10, 5)
    show_chunks = st.checkbox("Show retrieved passages", value=False)
    voice_mode = st.checkbox("🎙️ Voice mode", value=False)
    st.markdown("---")
    st.markdown(
        "**Model**: `qwen2.5:7b-instruct`  \n"
        "**Embeddings**: `bge-small-en-v1.5`  \n"
        "**Vector store**: FAISS"
    )

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🌙 Faith-Aware Wellness Companion")
st.caption(
    "A compassionate assistant grounded in Quranic verses and authenticated hadith. "
    "Not a substitute for professional mental health or medical advice."
)
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chunks") and show_chunks:
            with st.expander("📚 Retrieved passages"):
                for chunk in msg["chunks"]:
                    st.markdown(
                        f"**{chunk['reference']}** *(score: {chunk['score']:.3f})*  \n"
                        f"{chunk['text']}"
                    )
                    st.divider()


# ── Voice input helper ────────────────────────────────────────────────────────
def handle_voice_input() -> str | None:
    """Record from microphone, transcribe with faster-whisper, return text."""
    if not audio_recorder:
        st.error(
            "⚠️ Audio recorder not available. Please run: `pip install audio-recorder-streamlit`"
        )
        return None

    st.markdown("🎙️ **Click the mic to record, click again to stop:**")
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_size="2x",
        pause_threshold=3.0,   # auto-stop after 3 s of silence
        key="mic_recorder",
    )

    if audio_bytes:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        with st.spinner("Transcribing …"):
            from src.voice import transcribe
            text = transcribe(tmp_path)
        if text:
            st.success(f"📝 Heard: *{text}*")
            return text
        else:
            st.error("Transcription failed — please type your message instead.")
    return None


# ── Main input ────────────────────────────────────────────────────────────────
query: str | None = None

if voice_mode:
    query = handle_voice_input()
else:
    query = st.chat_input("How are you feeling? Ask anything …")


# ── Process query ─────────────────────────────────────────────────────────────
if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve + generate
    retriever = get_retriever()
    with st.spinner("Thinking …"):
        from src.generator import generate

        docs = retriever.search(query, top_k=top_k)
        response, latency_ms, tokens = generate(query, docs)

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(response)
        st.caption(f"⏱ {latency_ms:.0f} ms  |  tokens: {tokens or '—'}")
        if docs and show_chunks:
            with st.expander("📚 Retrieved passages"):
                for chunk in docs:
                    st.markdown(
                        f"**{chunk['reference']}** *(score: {chunk['score']:.3f})*  \n"
                        f"{chunk['text']}"
                    )
                    st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "chunks": docs}
    )

    # TTS if voice mode — speak response back
    if voice_mode:
        with st.spinner("🔊 Speaking response …"):
            from src.voice import tts_to_file, speak
            
            # Try to save to file for Streamlit playback
            audio_path = tts_to_file(response)
            
            if audio_path and audio_path.exists():
                # autoplay=True so it plays immediately without user clicking play
                st.audio(str(audio_path), format="audio/wav", autoplay=True)
            else:
                # Fallback: speak directly (no playback widget, but user hears it)
                st.info("🔊 Speaking… (using system audio)")
                speak(response)
