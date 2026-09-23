"""Very small JSONL/TF-IDF retrieval layer for historical report lessons."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def load_history(history_dir: str | Path) -> list[dict[str, Any]]:
    """Load all JSON objects, one per line, from every JSONL file."""
    path = Path(history_dir)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for file_path in path.glob("*.jsonl"):
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def search_similar_reports(query: str, history_dir: str | Path, top_k: int = 3) -> list[dict[str, Any]]:
    """Retrieve positive-score historical records using TF-IDF similarity."""
    records = load_history(history_dir)
    if not records:
        return []
    query_tokens = Counter(_tokens(query))
    documents = []
    for index, record in enumerate(records):
        text = json.dumps(record, ensure_ascii=False)
        tokens = Counter(_tokens(text))
        documents.append((index, tokens))

    total_documents = len(records)
    df_counts = Counter()
    for _, tokens in documents:
        df_counts.update(tokens.keys())
    scores: list[tuple[float, int]] = []
    for index, tokens in documents:
        score = 0.0
        for token, query_count in query_tokens.items():
            if token not in tokens:
                continue
            tf = tokens[token]
            idf = math.log((total_documents + 1) / (df_counts[token] + 1)) + 1
            score += tf * idf * query_count
        scores.append((score, index))
    scores.sort(reverse=True)
    return [records[index] for score, index in scores[:top_k] if score > 0]
