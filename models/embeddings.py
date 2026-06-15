"""
Embedding pipeline: encode up to 5000 fashion product images with CLIP ViT-B/32,
normalize vectors, and save embeddings + image ID list as artifacts.
"""

import os
import sys
import json
import time
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import faiss
import torch
import clip
from PIL import Image
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR   = os.path.join(ROOT, "data", "fashion-dataset", "images")
ARTIFACTS    = os.path.join(ROOT, "artifacts")
EMB_PATH     = os.path.join(ARTIFACTS, "embeddings.npy")
IDS_PATH     = os.path.join(ARTIFACTS, "image_ids.json")

MAX_IMAGES   = 30000
BATCH_SIZE   = 256   # larger batch on GPU
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_clip():
    log.info(f"Loading CLIP ViT-B/32 on {DEVICE} …")
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    log.info("CLIP loaded.")
    return model, preprocess


def collect_image_paths(images_dir: str, max_images: int) -> list[tuple[str, str]]:
    """Return list of (product_id, absolute_path) sorted by filename."""
    if not os.path.isdir(images_dir):
        log.error(f"Images directory not found: {images_dir}")
        sys.exit(1)

    entries: list[tuple[str, str]] = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        product_id = os.path.splitext(fname)[0]
        entries.append((product_id, os.path.join(images_dir, fname)))
        if len(entries) >= max_images:
            break

    log.info(f"Found {len(entries)} image files (cap={max_images}).")
    return entries


def encode_images(
    model,
    preprocess,
    entries: list[tuple[str, str]],
) -> tuple[np.ndarray, list[str]]:
    """
    Batch-encode images. Corrupted files are skipped silently.
    Returns (embeddings float32 N×512, image_ids list of str).
    """
    all_embeddings: list[np.ndarray] = []
    valid_ids: list[str] = []
    skipped = 0

    batches = [entries[i : i + BATCH_SIZE] for i in range(0, len(entries), BATCH_SIZE)]

    with torch.no_grad():
        for batch in tqdm(batches, desc="Encoding batches", unit="batch"):
            tensors: list[torch.Tensor] = []
            batch_ids: list[str] = []

            for product_id, path in batch:
                try:
                    img = Image.open(path).convert("RGB")
                    tensors.append(preprocess(img))
                    batch_ids.append(product_id)
                except Exception as exc:
                    log.warning(f"Skipping {path}: {exc}")
                    skipped += 1
                    continue

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors).to(DEVICE)          # (B, 3, 224, 224)
            features = model.encode_image(batch_tensor)              # (B, 512)
            features = features.cpu().numpy().astype(np.float32)     # numpy-2.x safe

            all_embeddings.append(features)
            valid_ids.extend(batch_ids)

    embeddings = np.concatenate(all_embeddings, axis=0)             # (N, 512)
    return embeddings, valid_ids, skipped


def normalize_and_save(embeddings: np.ndarray, image_ids: list[str]) -> None:
    os.makedirs(ARTIFACTS, exist_ok=True)

    # faiss.normalize_L2 requires float32 C-contiguous array
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    np.save(EMB_PATH, embeddings)
    log.info(f"Saved embeddings → {EMB_PATH}  shape={embeddings.shape}")

    with open(IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(image_ids, f)
    log.info(f"Saved image IDs  → {IDS_PATH}  count={len(image_ids)}")


def main() -> None:
    t0 = time.perf_counter()

    model, preprocess = load_clip()
    entries = collect_image_paths(IMAGES_DIR, MAX_IMAGES)

    if not entries:
        log.error("No images found. Download the Kaggle dataset first.")
        sys.exit(1)

    embeddings, image_ids, skipped = encode_images(model, preprocess, entries)
    normalize_and_save(embeddings, image_ids)

    elapsed = time.perf_counter() - t0
    log.info(
        f"Done. Processed={len(image_ids)}  Skipped={skipped}  "
        f"Elapsed={elapsed:.1f}s  ({len(image_ids)/elapsed:.1f} img/s)"
    )


if __name__ == "__main__":
    main()
