# Submission Email Note

---

**Subject:** Faith-Aware Wellness Companion — Take-Home Submission

Hi team,

Please find my submission linked below. The repository includes a working Streamlit prototype with live voice I/O, a FAISS RAG pipeline over the provided corpus, and a full evaluation harness covering retrieval recall, LLM-as-judge faithfulness, and adversarial safety probes.

---

**What I would build next with one more day:**

The biggest gap I'd close is **retrieval quality**. Right now articles are indexed as single ~300-word passages — splitting them into 100-word overlapping chunks with a sliding window would likely double recall@5 on article-dependent queries (q02, q05, q09 are all hurt by this). I'd pair that with **BM25 hybrid search** using reciprocal rank fusion, which would fix the keyword-exact hadith number queries that dense retrieval misses entirely.

Second, I'd replace `pyttsx3` with **Piper TTS** and a voice model trained on natural English speech. The robotic espeak output is the weakest part of the current demo and would undermine trust with a Muslim audience who deserves a warm, human-sounding voice.

Finally, I'd add **Ollama streaming** so the first tokens appear in ~2s instead of waiting 90–120s for the full response — the single biggest perceived latency win available without changing hardware.

---

Best regards
