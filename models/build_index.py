"""
Build a FAISS IndexFlatIP from pre-computed CLIP embeddings.
Vectors must already be L2-normalized (cosine similarity == dot product).
"""

import os
import time
import logging

import numpy as np
import faiss

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS  = os.path.join(ROOT, "artifacts")
EMB_PATH   = os.path.join(ARTIFACTS, "embeddings.npy")
INDEX_PATH = os.path.join(ARTIFACTS, "product_index.faiss")

EMBEDDING_DIM = 512

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    if not os.path.exists(EMB_PATH):
        log.error(f"Embeddings not found at {EMB_PATH}. Run embeddings.py first.")
        raise FileNotFoundError(EMB_PATH)

    log.info(f"Loading embeddings from {EMB_PATH} …")
    embeddings = np.load(EMB_PATH).astype(np.float32)
    embeddings = np.ascontiguousarray(embeddings)

    n, dim = embeddings.shape
    log.info(f"Embeddings shape: {n} × {dim}")

    if dim != EMBEDDING_DIM:
        raise ValueError(f"Expected dim={EMBEDDING_DIM}, got {dim}")

    log.info("Building FAISS IndexFlatIP …")
    t0 = time.perf_counter()

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    elapsed = time.perf_counter() - t0

    faiss.write_index(index, INDEX_PATH)

    log.info(f"Index built in {elapsed*1000:.1f} ms")
    log.info(f"Index size   : {index.ntotal} vectors")
    log.info(f"Saved index  → {INDEX_PATH}")
    log.info(f"File size    : {os.path.getsize(INDEX_PATH) / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
