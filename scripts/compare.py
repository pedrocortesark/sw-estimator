"""
CLI — similitud coseno entre dos textos usando text-embedding-3-small.

Uso:
    python scripts/compare.py --text-a "..." --text-b "..."

Dentro del contenedor:
    docker compose exec servicio_ia python scripts/compare.py --text-a "..." --text-b "..."

Fuera del contenedor (carga el .env automáticamente):
    uv run python scripts/compare.py --text-a "..." --text-b "..."
"""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.embedding_pipeline.embedder import OpenAIEmbedder  # noqa: E402


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calcula la similitud coseno entre dos textos usando OpenAI embeddings."
    )
    parser.add_argument("--text-a", required=True, help="Primer texto a comparar.")
    parser.add_argument("--text-b", required=True, help="Segundo texto a comparar.")
    args = parser.parse_args()

    embedder = OpenAIEmbedder()
    emb_a = embedder.embed_one(args.text_a)
    emb_b = embedder.embed_one(args.text_b)
    sim = cosine_similarity(emb_a, emb_b)

    print(f"Text A: {args.text_a}")
    print(f"Text B: {args.text_b}")
    print(f"Cosine similarity: {sim:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
