"""
Build corpus.jsonl files from raw sources.

Inputs (place these next to the script, or edit the paths below):
  - en.sahih.txt                  (from tanzil.net, Sahih International English)
  - Sahih al-Bukhari.json         (from huggingface.co/datasets/meeAtif/hadith_datasets)

Outputs (written into ../dataset/):
  - quran.jsonl
  - hadith.jsonl

Run:
  python build_corpus.py
"""

import json
import re
from pathlib import Path

# ---- paths ---------------------------------------------------------------
HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent / "dataset"
CORPUS.mkdir(exist_ok=True)

QURAN_SRC = HERE / "en.sahih.txt"
BUKHARI_SRC = HERE / "Bukhari.json"

# ---- curated selections --------------------------------------------------
# (surah, ayah, theme) — ~30 wellness-oriented verses
QURAN_PICKS = [
    (2, 152, "remembrance, gratitude"),
    (2, 153, "patience, prayer, hardship"),
    (2, 155, "patience, trials"),
    (2, 156, "grief, returning to Allah"),
    (2, 186, "closeness to Allah, du'a"),
    (2, 216, "trust, uncertainty"),
    (2, 286, "capacity, burden"),
    (3, 139, "hope, strength"),
    (3, 159, "tawakkul, reliance"),
    (3, 173, "tawakkul, sufficiency"),
    (6, 17, "trials, mercy"),
    (8, 46, "patience, perseverance"),
    (9, 51, "tawakkul, destiny"),
    (12, 87, "hope, despair"),
    (13, 11, "self-change, effort"),
    (13, 28, "peace of heart, remembrance"),
    (14, 7, "gratitude"),
    (16, 97, "good life, righteousness"),
    (20, 124, "heedlessness, constriction"),
    (24, 22, "forgiveness, mercy"),
    (29, 2, "trials, testing"),
    (39, 53, "despair, mercy, forgiveness"),
    (40, 60, "du'a, answered prayer"),
    (47, 7, "help from Allah"),
    (57, 22, "qadr, calamity"),
    (64, 11, "calamity, guidance of heart"),
    (65, 2, "way out, taqwa"),
    (65, 3, "provision, tawakkul"),
    (93, 3, "Allah has not abandoned you"),
    (94, 5, "ease with hardship"),
    (94, 6, "ease with hardship"),
]

# Bukhari hadith numbers to include (wellness / emotional / spiritual themes)
# Edit freely; script falls back gracefully if a number is missing.
BUKHARI_PICKS = [
    (1,    "intentions"),
    (10,   "Muslim defined"),
    (13,   "love for brother"),
    (39,   "ease of religion"),
    (52,   "doubtful matters, heart"),
    (660,  "shade of Allah, seven categories"),
    (1283, "patience at the first shock"),
    (1469, "contentment, self-restraint"),
    (3208, "provision, lifespan written"),
    (4837, "standing in prayer"),
    (5641, "reward for hardship"),
    (5645, "patience, calamity"),
    (5648, "patience, first strike"),
    (5671, "do not wish for death"),
    (5678, "cure for every disease"),
    (6018, "speak good or remain silent"),
    (6094, "truthfulness"),
    (6114, "anger, true strength"),
    (6116, "modesty"),
    (6306, "sayyid al-istighfar"),
    (6346, "du'a in distress"),
    (6369, "du'a for anxiety and grief"),
    (6407, "remembrance vs heedlessness"),
    (6412, "two blessings many neglect"),
    (6416, "live as a stranger"),
    (6470, "poverty of the heart"),
    (6490, "look at those below you"),
    (6502, "closeness through nawafil"),
    (7405, "Allah is as His servant thinks of Him"),
    (7510, "intercession and mercy"),
]

# ---- helpers -------------------------------------------------------------
def load_quran(path: Path) -> dict:
    """Tanzil format: `surah|ayah|text` per line. Returns {(s,a): text}."""
    out = {}
    if not path.exists():
        print(f"⚠️  Missing {path}. Skipping Quran build.")
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        s, a, text = parts
        try:
            out[(int(s), int(a))] = text.strip()
        except ValueError:
            continue
    return out


def load_bukhari(path: Path) -> list:
    """Load Bukhari JSON from meeAtif/hadith_datasets."""
    if not path.exists():
        print(f"⚠️  Missing {path}. Skipping Hadith build.")
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    # Dataset is a list of dicts; keys include Book, Chapter_Number,
    # Chapter_Title_English, Arabic_Text, English_Text, and a hadith number
    # field that varies by dump. We search for any integer-like field.
    return data


_REF_RE = re.compile(r"bukhari[:/](\d+)", re.IGNORECASE)

def _extract_ref_number(entry: dict) -> int | None:
    """Pull the global sunnah.com hadith number out of the Reference URL."""
    ref = entry.get("Reference") or entry.get("reference") or ""
    m = _REF_RE.search(str(ref))
    return int(m.group(1)) if m else None


def find_hadith(entries: list, number: int) -> dict | None:
    """Find a hadith by its sunnah.com reference number."""
    for entry in entries:
        n = _extract_ref_number(entry)
        if n == number:
            return entry
    return None


def clean_text(s: str) -> str:
    """Collapse whitespace, strip HTML-ish remnants."""
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---- build quran.jsonl ---------------------------------------------------
def build_quran():
    verses = load_quran(QURAN_SRC)
    if not verses:
        return
    out_path = CORPUS / "quran.jsonl"
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for (s, a, theme) in QURAN_PICKS:
            text = verses.get((s, a))
            if not text:
                print(f"   missing Quran {s}:{a}")
                continue
            rec = {
                "id": f"quran_{s}_{a}",
                "source": "Quran",
                "reference": f"Surah {s}:{a}",
                "surah": s,
                "ayah": a,
                "theme": theme,
                "text": clean_text(text),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"✅ quran.jsonl — {n} verses")


# ---- build hadith.jsonl --------------------------------------------------
def build_hadith():
    entries = load_bukhari(BUKHARI_SRC)
    if not entries:
        return
    out_path = CORPUS / "hadith.jsonl"
    n = 0
    seen = set()
    with out_path.open("w", encoding="utf-8") as f:
        for (num, theme) in BUKHARI_PICKS:
            if num in seen:
                continue
            seen.add(num)
            entry = find_hadith(entries, num)
            if not entry:
                print(f"   missing Bukhari #{num}")
                continue
            eng = clean_text(entry.get("English_Text", ""))
            if not eng or len(eng) < 20:
                continue
            rec = {
                "id": f"hadith_bukhari_{num}",
                "source": "Sahih al-Bukhari",
                "reference": f"Sahih al-Bukhari {num}",
                "number": num,
                "chapter": entry.get("Chapter_Title_English", ""),
                "theme": theme,
                "text": eng,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"✅ hadith.jsonl — {n} hadith")


if __name__ == "__main__":
    print("Building corpus…")
    build_quran()
    build_hadith()
    print("Done. Output in:", CORPUS)
