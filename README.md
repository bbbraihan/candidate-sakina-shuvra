# Take-Home Project: Faith-Aware Wellness Companion

## Background

We are building AI-powered tools to support the personal and spiritual well-being of Muslims navigating stress, identity, and questions of faith. This take-home is a scoped prototype that mirrors the kind of work you would do on the team: an LLM-powered assistant grounded in trustworthy Islamic sources, evaluated rigorously, and designed with production realities in mind.

We expect this to take **2–4 hours**. Use of Cursor, Claude, Copilot, or any other tooling is encouraged — we evaluate engineering judgment, not typing speed. Please do not spend more than 4 hours; we would rather see a smaller, sharper submission than a sprawling one.

---

## What this starter package contains

```
starter_package/
├── README.md             ← this file
├── SETUP.md              ← setup guide (read first)
├── corpus/
│   ├── quran.jsonl       ← ~30 curated verses (Sahih International)
│   ├── hadith.jsonl      ← ~30 hadith from Sahih al-Bukhari
│   ├── articles.jsonl    ← 5 original wellness articles
│   └── labels.json       ← 10 labeled queries for retrieval eval
└── scripts/
    └── build_corpus.py   ← how the corpus was assembled (for reference)
```

Start by reading **`SETUP.md`**.

---

## ⚠️ Model constraints

**LLM:** Choose any instruction-tuned, openly licensed LLM **≤ 8B parameters** available on Hugging Face that you can run locally. Good options include Qwen 2.5 7B Instruct, Llama 3.1 8B Instruct, Mistral 7B Instruct, Gemma 2 9B Instruct, or Phi-3.5 Mini. In your README, state the **exact** model, source repo, and quantization (e.g., `bartowski/Qwen2.5-7B-Instruct-GGUF`, `Q4_K_M`) and the exact command you used to run it. We will reproduce your eval numbers using this model, so reproducibility information is not optional.

**Embedding model:** Use `BAAI/bge-small-en-v1.5`. This is fixed so that retrieval results are comparable across submissions.

**No API-based LLMs.** Do not use OpenAI, Anthropic, Cohere, Google, or any paid API for generation, embeddings, or evaluation. Everything must run locally from open weights. Submissions using API models will not be evaluated.

Model selection is part of what we are evaluating — explain in your README **why** you chose the model you did. A Lead AI/ML Engineer should be able to extract strong behavior from a constrained ≤8B model through good prompting, retrieval, and evaluation, not by reaching for a frontier API.

---

## Deliverables

A GitHub repository (or zip) containing:

1. **Working prototype** — CLI, FastAPI endpoint, or a minimal Streamlit/Gradio UI. Your choice; justify it briefly.
2. **README** with setup instructions, an architecture diagram (a hand sketch is fine), key design decisions, model choice rationale, and tradeoffs you made under the time budget.
3. **`EVAL.md`** — your evaluation report with actual numbers.
4. **Optional 3–5 minute Loom walkthrough.** Helpful but not required.

---

## Required functionality

### 1. RAG over the provided corpus
- Build an embedding index using a vector store of your choice (FAISS, Chroma, pgvector, LanceDB, etc.). Justify the choice in your README.
- Use the fixed `bge-small-en-v1.5` embedding model.
- Retrieve top-k passages relevant to the user's emotional or spiritual query.
- Each corpus record has an `id` and a `reference` field — use them for citations.

### 2. LLM-powered response generation
- Use your chosen local LLM (see Model constraints above).
- Your system prompt must produce responses that:
  - (a) are empathetic and non-judgmental,
  - (b) cite retrieved sources using their `reference` (e.g., *Surah Ar-Ra'd 13:28*, *Sahih al-Bukhari 6369*),
  - (c) **never fabricate verses or hadith**,
  - (d) avoid issuing medical advice or fatwa-level rulings, deferring appropriately.

### 3. Voice I/O
Use **fully local, free** components — no paid APIs:
- **Speech-to-text**: `faster-whisper` or `openai-whisper` (local), or the browser Web Speech API.
- **Text-to-speech**: Piper, Coqui TTS, or `pyttsx3` (all local and free).
- Must work end-to-end: user speaks → assistant speaks back.

### 4. Evaluation framework (the differentiator)
Build a small but real eval harness with **at least 10 test cases** covering:
- **Retrieval quality** — recall@k against the provided `corpus/labels.json` (we recommend reporting recall@3 and recall@5).
- **Faithfulness / no-hallucination** — does the response only cite passages that were actually retrieved? Using your local LLM itself as the judge is acceptable; document the rubric. Manual labels for a few cases are also welcome.
- **Safety** — adversarial prompts including suicidal ideation, requests for fatwa, sectarian bait, and prompt-injection attempts. Define expected behavior for each and score.
- **Latency** — p50/p95 for retrieval, generation, and the end-to-end voice round-trip on your hardware (state your specs).

Output a results table in `EVAL.md`.

### 5. Production-mindedness
- Configuration via environment variables; no hardcoded paths or secrets.
- Structured logging of every request: prompt, retrieved chunk IDs, response, latency, and token count.
- Graceful fallback when STT, TTS, or the LLM fails.

---

## Out of scope

Authentication, polished UI, cloud deployment, model training or fine-tuning. We want depth on **RAG, evaluation, and safety**, not breadth.

---

## How we will evaluate

| Area | Weight | What "strong" looks like |
|---|---|---|
| RAG design | 20% | Sensible chunking, justified retrieval strategy, considered hybrid search or reranking, citations tied to retrieved IDs |
| Evaluation framework | 25% | Real metrics with numbers. Retrieval and generation evaluated separately. Adversarial set included. |
| Safety & domain handling | 20% | Crisis routing (e.g., suicidal ideation → helpline), no fabricated scripture, refuses to issue fatwa, culturally aware tone |
| Voice integration | 10% | Reliable round-trip, error handling, reasonable latency |
| Code quality & structure | 10% | Clear modules (retriever / generator / evaluator / voice), typed, testable |
| Production thinking | 10% | Logging, configs, latency awareness, observable failures |
| Communication | 5% | README explains *why*, not just *what*; tradeoffs and model choice called out explicitly |

---

## Submission

Send a link to the repo (or a zip) plus your `EVAL.md` and any video walkthrough. In the email, please include a short note (≤200 words) on what you would build next if you had another full day.

Your submission must include **clear instructions for us to reproduce your evaluation locally**, including the exact model name, quantization, and the command to pull or run it.

---

## A few practical tips

- **Start with the eval harness, not the chatbot.** If you build retrieval and evaluation first, every later change is measurable. Candidates who build the full pipeline and *then* try to evaluate it usually run out of time.
- **A 7B–8B model will surprise you on quality if you prompt it well.** Few-shot examples, structured output (JSON with a `citations` field), and clear system-prompt constraints matter more than model size at this scale.
- **Cache embeddings to disk.** Recomputing them on every startup will eat your time budget.
- **Be honest in `EVAL.md`.** Failure modes you identified and didn't have time to fix are worth more than glossed-over numbers. We read these carefully.

Good luck — we are looking forward to seeing how you think.
