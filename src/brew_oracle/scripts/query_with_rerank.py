from __future__ import annotations

import argparse

from sentence_transformers import CrossEncoder

from brew_oracle.utils.config import Settings
from brew_oracle.knowledge.pdf_kb import build_pdf_kb


def main() -> None:
    parser = argparse.ArgumentParser(description="Query PDF KB with rerank")
    parser.add_argument("query", nargs="?", help="Pergunta a ser pesquisada")
    args = parser.parse_args()

    query = args.query or input("Pergunta: ")

    s = Settings()
    kb = build_pdf_kb()

    docs = kb.search(query, top_k=s.TOP_K)

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [
        (query, getattr(doc, "content", getattr(doc, "text", "")))
        for doc in docs
    ]
    scores = cross_encoder.predict(pairs)

    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    for idx, (doc, score) in enumerate(reranked[:5], 1):
        meta = getattr(doc, "meta", {}) or getattr(doc, "metadata", {})
        source = meta.get("source") or meta.get("file_path") or "?"
        page = meta.get("page_number") or meta.get("page") or "?"
        snippet = (getattr(doc, "content", getattr(doc, "text", ""))).strip().replace("\n", " ")
        print(f"{idx}. score={score:.4f} | {source} p.{page}: {snippet}")


if __name__ == "__main__":
    main()
