# SETUP

This guide gets you from zero to a running local LLM + embedding model in about 10 minutes. You **choose the LLM** (see constraints in `README.md`); the embedding model is fixed.

---

## 1. System requirements

- **RAM:** 8 GB minimum free (a 7B–8B Q4_K_M model occupies ~5 GB in memory).
- **Disk:** ~6 GB free for the model weights.
- **OS:** macOS, Linux, or Windows with WSL2.
- **Python:** 3.10 or newer.

Any mid-range laptop from the last 5 years will handle this. Ollama uses Metal on Apple Silicon and CUDA on NVIDIA automatically if available, and falls back to CPU cleanly.

---

## 2. Install Ollama (recommended LLM runtime)

**macOS / Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:** download the installer from https://ollama.com/download.

Verify:
```bash
ollama --version
```

You are free to use `llama.cpp`, `vLLM`, or `transformers` directly if you prefer — Ollama is just the fastest path.

---

## 3. Pull your chosen LLM

Pick **one** model ≤ 8B parameters. Good options (any one is fine):

```bash
# Qwen 2.5 7B Instruct (strong RAG behavior, Apache 2.0)
ollama pull qwen2.5:7b-instruct

# Llama 3.1 8B Instruct (Meta community license)
ollama pull llama3.1:8b-instruct-q4_K_M

# Mistral 7B Instruct (Apache 2.0)
ollama pull mistral:7b-instruct

# Gemma 2 9B Instruct (borderline at 9B, allowed)
ollama pull gemma2:9b-instruct-q4_K_M

# Phi-3.5 Mini (3.8B, runs on low-RAM machines, MIT)
ollama pull phi3.5:3.8b-mini-instruct
```

Verify:
```bash
ollama run qwen2.5:7b-instruct "Say hello in one sentence."
```

You should see a coherent reply. Press `Ctrl+D` to exit.

**In your README, record the exact model tag you used** — e.g., `qwen2.5:7b-instruct` at `Q4_K_M`. We will pull the same tag to reproduce your numbers.

---

## 4. Use the model from Python

Ollama exposes an **OpenAI-compatible** endpoint at `http://localhost:11434/v1`, so you can use the standard `openai` SDK with no code changes beyond the base URL:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # any non-empty string; Ollama ignores it
)

resp = client.chat.completions.create(
    model="qwen2.5:7b-instruct",  # whatever you pulled
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Name two Quranic themes related to patience."},
    ],
    temperature=0.3,
)
print(resp.choices[0].message.content)
```

LangChain, LlamaIndex, and every other major framework accept this pattern.

---

## 5. Load the fixed embedding model

Everyone uses the same embedding model so retrieval numbers are comparable.

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

# First run auto-downloads from Hugging Face (~130 MB) and caches locally.
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
vectors = embedder.encode(["Hearts find peace in the remembrance of Allah."])
print(vectors.shape)  # (1, 384)
```

This is a 384-dimension English embedding model. It runs on CPU in milliseconds.

---

## 6. Corpus files

Under `corpus/` you will find:

- `quran.jsonl` — ~30 curated verses (Sahih International translation).
- `hadith.jsonl` — ~30 hadith from Sahih al-Bukhari.
- `articles.jsonl` — 5 original wellness articles.
- `labels.json` — 10 labeled queries for the retrieval eval (recall@k).

Each corpus record has at minimum: `id`, `source`, `reference`, `theme`, and `text`. Use `id` for citations and for the retrieval eval.

---

## 7. Troubleshooting

**Model replies are extremely slow (> 30s per short reply)**
You are likely running on CPU only. That is fine — plan your eval runs accordingly, keep prompts short, and cache aggressively. Report the specs you used in `EVAL.md`.

**Out-of-memory errors**
Close other applications, or switch to a smaller model (`phi3.5:3.8b-mini-instruct` runs on 6 GB RAM). Record the change in your README.

**`ollama pull` is slow**
Models are 3–5 GB. On a typical home connection this takes 5–10 minutes. Do this before you start your 2–4 hour timer.

**Anything else**
Document the issue in your README and move on. A clear note about a workaround is worth more than a silent hack.
