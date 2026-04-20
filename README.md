# Faith-Aware Wellness Companion

A locally-running RAG chatbot that provides compassionate, Islam-grounded emotional
and spiritual support — built for the Girmairi take-home evaluation.

**Prototype choice:** Streamlit UI — fastest path to a demo-able, video-recordable
interface. A FastAPI backend would be cleaner for production; Streamlit is the right
call for a 4-hour scoped prototype.

---

## Architecture

```
User (text or voice)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                   Streamlit UI  (app.py)               │
│  ┌──────────────────┐    ┌────────────────────────┐   │
│  │ Voice mode        │    │ Text chat mode          │   │
│  │ audio_recorder    │    │ st.chat_input           │   │
│  │ → faster-whisper  │    │                         │   │
│  │ → pyttsx3 TTS     │    │                         │   │
│  └────────┬──────────┘    └──────────┬──────────────┘  │
└───────────┼──────────────────────────┼──────────────────┘
            │           query           │
            ▼                           ▼
┌────────────────────────────────────────────────┐
│                src/retriever.py                 │
│  FAISS IndexFlatIP  (cosine via inner product)  │
│  66 passages: Quran (31) + Hadith (30) +        │
│  Articles (5), embedded with bge-small-en-v1.5  │
│  Embedding text = theme + passage text          │
│  Query prefix = wellness instruction template   │
│  Cached to faiss_index/ on first run            │
└──────────────────────┬─────────────────────────┘
                       │  top-k docs (id + reference + text)
                       ▼
┌────────────────────────────────────────────────┐
│                src/generator.py                 │
│  1. Python crisis check → crisis line (hard)   │
│  2. Ollama → qwen2.5:7b-instruct               │
│  3. System prompt: empathy + NO fabrication +  │
│     fatwa/medical deferral + sectarian neutral │
│  4. Response with inline *reference* citations │
└──────────────────────┬─────────────────────────┘
                       │  response + latency + tokens
                       ▼
              src/logger.py  →  logs/requests.jsonl
```

---

## Setup

### Prerequisites

- Python 3.12
- [Ollama](https://ollama.com) installed and running
- `qwen2.5:7b-instruct` already pulled: `ollama pull qwen2.5:7b-instruct`
- `espeak` for TTS (Linux): `sudo apt install espeak`

### Install

```bash
git clone <repo-url>
cd candidate-sakina-shuvra

pip install -r requirements.txt

# Optional: copy env config
cp .env.example .env
```

### Run the app

```bash
streamlit run app.py
```

### Run evaluation

```bash
python eval/run_all.py
```

Outputs `EVAL.md` with all metrics.

---

## Model Choice Rationale

| Component    | Choice                           | Why                                                                                                                                                                              |
| ------------ | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLM          | `qwen2.5:7b-instruct` via Ollama | Strong multi-rule instruction following at 7B; top of open-weight 7B benchmarks; Ollama's OpenAI-compatible API makes it trivial to swap. Default Ollama quantisation is Q4_K_M. |
| Embedding    | `BAAI/bge-small-en-v1.5`         | Fixed by spec. 33M params, strong retrieval quality, fast on CPU.                                                                                                                |
| Vector store | FAISS `IndexFlatIP`              | No server process; single-file cache; exact cosine search via inner product on L2-normalised vectors; 66-doc corpus fits in RAM.                                                 |
| STT          | `faster-whisper` small           | Best local accuracy/speed on CPU; int8 quantisation; forced `language="en"` prevents mistranscription.                                                                           |
| TTS          | `pyttsx3` + espeak               | Zero extra model download; fully offline; graceful fallback to system audio if WAV generation fails.                                                                             |
| UI           | Streamlit                        | Fastest to demo; `audio_recorder_streamlit` provides browser mic capture; `st.audio(autoplay=True)` closes the voice loop.                                                       |

**Exact model command to reproduce:**

```bash
ollama pull qwen2.5:7b-instruct
ollama run qwen2.5:7b-instruct
```

### Why not reranking?

With 66 documents a cross-encoder adds ~500ms latency with minimal recall gain. Justified at 10k+ docs.

### Retrieval strategy details

- **Embedding enrichment**: passage text is prefixed with its `theme` tag at index time (e.g. _"patience, trials. And We will surely test you…"_) to boost thematic semantic matching.
- **Query prefix**: queries are prefixed with `"Represent this wellness question for searching relevant Islamic guidance: …"` — a known trick to improve bge-small recall by aligning query/document spaces.
- **top-k=5 default**: gives recall@5 margin while keeping context window manageable.

---

## Key Design Decisions

**Safety is a hard-stop, not a prompt suggestion.**
Suicidal ideation detection runs in Python _before_ the LLM call (`generator.py:_is_crisis()`). No prompt injection or jailbreak can bypass it — the LLM is never even invoked.

**No fabrication by construction.**
The system prompt explicitly forbids citing anything outside the `CONTEXT` block and instructs the model to say "I don't have a source for this" rather than hallucinate. A faithfulness judge (same LLM) audits this in eval.

**Culturally aware, non-judgmental tone.**
The system prompt is written for a Muslim audience: it uses the correct transliterations (_tawakkul_, _du'a_, _dhikr_), avoids imposing any single school of thought, and defers fatwa-level questions to qualified scholars — not to silence the question but to respect the weight of Islamic jurisprudence.

**Embeddings cached to disk.**
`faiss_index/corpus.index` + `corpus_meta.pkl` written on first run. Subsequent starts are instant.

**Structured logging on every request.**
`logs/requests.jsonl` — query, retrieved chunk IDs, response snippet, latency, token count. Fully observable without a metrics server.

**Graceful degradation.**

- LLM timeout/error → user-facing apology message, error logged.
- STT failure → text input shown, error surfaced in UI.
- TTS WAV empty → falls back to `pyttsx3.speak()` on system audio.
- All failures caught with `try/except`, logged at WARNING level.

---

## Voice I/O Round-Trip

1. User clicks mic button (`audio_recorder_streamlit`) → WAV bytes captured in browser
2. Saved to temp file → `faster-whisper` STT with `language="en"` forced
3. Transcribed text shown to user for confirmation
4. RAG pipeline runs (retrieval + generation)
5. Response displayed as text + spoken back via `pyttsx3` WAV → `st.audio(autoplay=True)`

---

## Tradeoffs Under Time Budget

| Tradeoff              | Decision                                                                                  |
| --------------------- | ----------------------------------------------------------------------------------------- |
| No BM25 hybrid search | Dense-only sufficient for 66 docs; BM25 would help on keyword queries like "Bukhari 6369" |
| pyttsx3 over Piper    | Piper quality is better but needs model download; pyttsx3 + espeak works out of the box   |
| No chunk splitting    | Articles ~300 words each — acceptable as single passages; splitting would help recall at  |

logs/ # auto-generated request log

```

```
