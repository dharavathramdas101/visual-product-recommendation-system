"""
FastAPI backend for the Visual Product Recommendation System.
Loads CLIP, FAISS index, image IDs, and product metadata at startup.
"""

import os
import sys
import io
import json
import time
import logging
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
import faiss
import torch
import clip
import pandas as pd
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── project root on sys.path so utils/ is importable ───────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.mlflow_logger import MLflowLogger

# ── paths ──────────────────────────────────────────────────────────────────────
ARTIFACTS   = os.path.join(ROOT, "artifacts")
EMB_PATH    = os.path.join(ARTIFACTS, "embeddings.npy")
INDEX_PATH  = os.path.join(ARTIFACTS, "product_index.faiss")
IDS_PATH    = os.path.join(ARTIFACTS, "image_ids.json")
STYLES_PATH = os.path.join(ROOT, "data", "fashion-dataset", "styles.csv")
IMAGES_DIR  = os.path.join(ROOT, "data", "fashion-dataset", "images")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── global state (populated in lifespan) ───────────────────────────────────────
state: dict = {}


# ── lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading CLIP ViT-B/32 …")
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    state["model"]      = model
    state["preprocess"] = preprocess
    log.info(f"CLIP loaded on {DEVICE.upper()}.")

    # GPU warmup — eliminates cold-start latency on first user request
    if DEVICE == "cuda":
        log.info("Warming up GPU …")
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=DEVICE)
            model.encode_image(dummy)
        log.info("GPU warm.")

    log.info("Loading FAISS index …")
    if not os.path.exists(INDEX_PATH):
        log.error(f"FAISS index not found: {INDEX_PATH}. Run build_index.py first.")
        raise RuntimeError("FAISS index missing.")
    index = faiss.read_index(INDEX_PATH)
    state["index"] = index
    log.info(f"FAISS index loaded: {index.ntotal} vectors.")

    log.info("Loading image IDs …")
    with open(IDS_PATH, "r", encoding="utf-8") as f:
        image_ids: list[str] = json.load(f)
    state["image_ids"] = image_ids

    log.info("Loading product metadata …")
    df = pd.read_csv(
        STYLES_PATH,
        dtype={"id": str},
        on_bad_lines="skip",
    )
    df = df.set_index("id")
    state["metadata"] = df
    log.info(f"Metadata loaded: {len(df)} products.")

    logger = MLflowLogger()
    logger.log_startup(
        dataset_size=index.ntotal,
        model_name="CLIP ViT-B/32",
        index_type="IndexFlatIP",
        embedding_dim=512,
    )
    state["logger"] = logger

    yield

    logger.end_run()
    log.info("Shutdown complete.")


# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Visual Product Recommender",
    version="1.0.0",
    lifespan=lifespan,
)

if os.path.isdir(IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

MYNTRA_DIR = os.path.join(ROOT, "data", "fashion-dataset", "myntradataset", "images")
if os.path.isdir(MYNTRA_DIR):
    app.mount("/myntradataset/images", StaticFiles(directory=MYNTRA_DIR), name="myntra_images")


# ── schemas ────────────────────────────────────────────────────────────────────
class ProductResult(BaseModel):
    product_id:           str
    product_name:         str
    image_path:           str
    category:             str
    gender:               str
    similarity_score:     float
    inference_latency_ms: float


# ── helpers ────────────────────────────────────────────────────────────────────
def _encode_image(img: Image.Image) -> np.ndarray:
    preprocess = state["preprocess"]
    model      = state["model"]
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model.encode_image(tensor)
    vec = features.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vec)
    return vec                          # shape (1, 512)


def _encode_text(text: str) -> np.ndarray:
    model = state["model"]
    tokens = clip.tokenize([text]).to(DEVICE)
    with torch.no_grad():
        features = model.encode_text(tokens)
    vec = features.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vec)
    return vec                          # shape (1, 512)


def _search(
    query_vec:  np.ndarray,
    top_k:      int,
    latency_ms: float,
    gender:     str = "All",
) -> list[ProductResult]:
    index     = state["index"]
    image_ids = state["image_ids"]
    df        = state["metadata"]

    # over-fetch when filtering so we still return top_k after gender filter
    fetch_k = top_k * 6 if gender != "All" else top_k
    fetch_k = min(fetch_k, index.ntotal)

    scores, indices = index.search(query_vec, fetch_k)

    results: list[ProductResult] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(image_ids):
            continue
        pid = image_ids[idx]

        if pid in df.index:
            row           = df.loc[pid]
            category      = str(row.get("masterCategory", "Unknown"))
            name          = str(row.get("productDisplayName", pid))
            product_gender = str(row.get("gender", "Unisex"))
        else:
            category       = "Unknown"
            name           = pid
            product_gender = "Unisex"

        # gender filter
        if gender != "All" and product_gender.lower() != gender.lower():
            continue

        results.append(
            ProductResult(
                product_id=pid,
                product_name=name,
                image_path=f"images/{pid}.jpg",
                category=category,
                gender=product_gender,
                similarity_score=float(score),
                inference_latency_ms=latency_ms,
            )
        )
        if len(results) >= top_k:
            break

    return results


def _log(query_type: str, top_k: int, latency_ms: float, result_count: int) -> None:
    try:
        state["logger"].log_request(
            query_type=query_type,
            top_k=top_k,
            latency_ms=latency_ms,
            result_count=result_count,
        )
    except Exception as exc:
        log.warning(f"MLflow logging failed: {exc}")


# ── endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "status":       "ok",
            "model_loaded": "model" in state,
            "index_size":   state["index"].ntotal if "index" in state else 0,
            "device":       DEVICE,
        }
    )


@app.post("/recommend/image", response_model=list[ProductResult])
async def recommend_image(
    file:   UploadFile = File(...),
    top_k:  int        = Query(default=10, ge=1, le=50),
    gender: str        = Query(default="All"),
):
    t0 = time.perf_counter()

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, Exception) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    try:
        query_vec = _encode_image(img)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {exc}")

    latency_ms = (time.perf_counter() - t0) * 1000
    results    = _search(query_vec, top_k, latency_ms, gender)
    _log("image", top_k, latency_ms, len(results))

    log.info(f"[image]  top_k={top_k}  gender={gender}  latency={latency_ms:.1f}ms  results={len(results)}")
    return results


@app.post("/recommend/text", response_model=list[ProductResult])
async def recommend_text(
    query:  str = Form(...),
    top_k:  int = Form(default=10, ge=1, le=50),
    gender: str = Form(default="All"),
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    t0 = time.perf_counter()

    try:
        query_vec = _encode_text(query.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {exc}")

    latency_ms = (time.perf_counter() - t0) * 1000
    results    = _search(query_vec, top_k, latency_ms, gender)
    _log("text", top_k, latency_ms, len(results))

    log.info(f"[text]   top_k={top_k}  gender={gender}  latency={latency_ms:.1f}ms  query='{query}'")
    return results


@app.post("/recommend/hybrid", response_model=list[ProductResult])
async def recommend_hybrid(
    file:   UploadFile = File(...),
    query:  str        = Form(...),
    top_k:  int        = Form(default=10, ge=1, le=50),
    alpha:  float      = Form(default=0.6),
    gender: str        = Form(default="All"),
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if not (0.0 <= alpha <= 1.0):
        raise HTTPException(status_code=400, detail="alpha must be in [0, 1]")

    t0 = time.perf_counter()

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, Exception) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    try:
        img_vec  = _encode_image(img)            # (1, 512) normalized
        txt_vec  = _encode_text(query.strip())   # (1, 512) normalized
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {exc}")

    # weighted fusion then re-normalize
    fused = alpha * img_vec + (1.0 - alpha) * txt_vec
    faiss.normalize_L2(fused)

    latency_ms = (time.perf_counter() - t0) * 1000
    results    = _search(fused, top_k, latency_ms, gender)
    _log("hybrid", top_k, latency_ms, len(results))

    log.info(
        f"[hybrid] top_k={top_k}  gender={gender}  latency={latency_ms:.1f}ms  "
        f"alpha={alpha}  query='{query}'"
    )
    return results


@app.get("/product/{product_id}")
def get_product(product_id: str) -> JSONResponse:
    df = state.get("metadata")
    if df is None:
        raise HTTPException(status_code=503, detail="Metadata not loaded.")

    if product_id not in df.index:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found.")

    row = df.loc[product_id]
    return JSONResponse(
        {
            "product_id":         product_id,
            "productDisplayName": str(row.get("productDisplayName", "")),
            "masterCategory":     str(row.get("masterCategory", "")),
            "subCategory":        str(row.get("subCategory", "")),
            "articleType":        str(row.get("articleType", "")),
            "baseColour":         str(row.get("baseColour", "")),
            "season":             str(row.get("season", "")),
            "year":               str(row.get("year", "")),
            "usage":              str(row.get("usage", "")),
            "image_path":         f"images/{product_id}.jpg",
        }
    )
