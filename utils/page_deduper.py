import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


PAGE_BREAK = "<!-- page break -->"


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.strip()


def split_pages(markdown: str) -> List[str]:
    pages = markdown.split(PAGE_BREAK)
    return [p.strip() for p in pages if p.strip()]


def hash_page(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def detect_duplicates(markdown_text: str, embedder):

    pages = split_pages(markdown_text)
    normalized = [normalize(p) for p in pages]

    results = []

    # ---------- EXACT DUPLICATES ----------
    hash_map = {}

    for i, p in enumerate(normalized):
        h = hash_page(p)
        hash_map.setdefault(h, []).append(i + 1)

    for cluster in hash_map.values():
        if len(cluster) > 1:
            results.append({
                "pages": cluster,
                "score": 1.0
            })

    # ---------- NEAR DUPLICATES ----------
    embeddings = embedder.embed_documents(normalized)

    n = len(embeddings)

    for i in range(n):
        for j in range(i + 1, n):

            len_ratio = len(normalized[i]) / max(len(normalized[j]), 1)

            if not (0.8 <= len_ratio <= 1.25):
                continue

            sim = cosine_similarity(
                [embeddings[i]], [embeddings[j]]
            )[0][0]

            if sim >= 0.9:
                results.append({
                    "pages": [i + 1, j + 1],
                    "score": float(sim)
                })

    return {
        "total_pages": len(pages),
        "duplicates": results
    }

def save_duplicate_json(result: Dict[str, Any], output_folder: Path, stem: str):

    E_dir = output_folder / "E"
    E_dir.mkdir(exist_ok=True)

    out = E_dir / f"{stem}_page_duplicates.json"

    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)