"""
Evaluation script for the Visual Product Recommender.
Runs 10 text + 10 image queries, measures:
  - Precision@10 (proxied by category consistency since no ground truth)
  - Category consistency @ 10
  - Average inference latency (ms)

Requires the FastAPI server to be running at API_URL.
"""

import os
import sys
import json
import time
import logging

import requests
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STYLES_CSV = os.path.join(ROOT, "data", "fashion-dataset", "styles.csv")
IMAGES_DIR = os.path.join(ROOT, "data", "fashion-dataset", "images")
OUT_PATH   = os.path.join(ROOT, "evaluation", "evaluation_results.json")

API_URL = os.getenv("API_URL", "http://localhost:8000")
TOP_K   = 10

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── text queries with expected category ────────────────────────────────────────
TEXT_QUERIES: list[dict] = [
    {"query": "red summer dress",             "expected_category": "Apparel"},
    {"query": "blue denim jeans",             "expected_category": "Apparel"},
    {"query": "white sneakers running shoes", "expected_category": "Footwear"},
    {"query": "black leather handbag",        "expected_category": "Accessories"},
    {"query": "casual cotton t-shirt",        "expected_category": "Apparel"},
    {"query": "formal men suit jacket",       "expected_category": "Apparel"},
    {"query": "sunglasses UV protection",     "expected_category": "Accessories"},
    {"query": "sports gym shorts",            "expected_category": "Apparel"},
    {"query": "women kurta ethnic wear",      "expected_category": "Apparel"},
    {"query": "flip flops sandals beach",     "expected_category": "Footwear"},
]


def load_metadata() -> pd.DataFrame:
    if not os.path.exists(STYLES_CSV):
        log.error(f"styles.csv not found: {STYLES_CSV}")
        sys.exit(1)
    df = pd.read_csv(STYLES_CSV, dtype={"id": str}, on_bad_lines="skip")
    return df.set_index("id")


def pick_image_queries(df: pd.DataFrame) -> list[dict]:
    """
    Pick one representative image per major category from the indexed images.
    Falls back to whatever images exist if categories are sparse.
    """
    target_categories = [
        "Apparel", "Footwear", "Accessories", "Personal Care",
        "Sporting Goods", "Free Items", "Home", "Apparel",
        "Footwear", "Accessories",
    ]
    queries: list[dict] = []
    used_ids: set[str] = set()

    for cat in target_categories:
        candidates = df[df["masterCategory"] == cat] if "masterCategory" in df.columns else df
        for pid in candidates.index:
            if pid in used_ids:
                continue
            img_path = os.path.join(IMAGES_DIR, f"{pid}.jpg")
            if os.path.exists(img_path):
                queries.append({"product_id": pid, "image_path": img_path, "expected_category": cat})
                used_ids.add(pid)
                break

    # fill up to 10 with any available image if needed
    if len(queries) < 10:
        for fname in sorted(os.listdir(IMAGES_DIR)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            pid = os.path.splitext(fname)[0]
            if pid in used_ids:
                continue
            cat = "Unknown"
            if pid in df.index and "masterCategory" in df.columns:
                cat = str(df.loc[pid, "masterCategory"])
            queries.append(
                {"product_id": pid, "image_path": os.path.join(IMAGES_DIR, fname), "expected_category": cat}
            )
            used_ids.add(pid)
            if len(queries) >= 10:
                break

    return queries[:10]


def category_consistency(results: list[dict], expected_category: str) -> float:
    if not results or expected_category == "Unknown":
        return 0.0
    matches = sum(1 for r in results if r.get("category", "") == expected_category)
    return matches / len(results)


def run_text_query(q: dict) -> dict:
    try:
        t0 = time.perf_counter()
        resp = requests.post(
            f"{API_URL}/recommend/text",
            data={"query": q["query"], "top_k": TOP_K},
            timeout=30,
        )
        wall_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            log.warning(f"Text query failed [{resp.status_code}]: {q['query']}")
            return _failed_row("text", q["query"])

        results   = resp.json()
        infer_ms  = results[0]["inference_latency_ms"] if results else wall_ms
        cat_score = category_consistency(results, q["expected_category"])

        return {
            "type":                 "text",
            "query":                q["query"],
            "expected_category":    q["expected_category"],
            "precision_at_10":      cat_score,
            "category_consistency": cat_score,
            "latency_ms":           infer_ms,
            "result_count":         len(results),
        }
    except requests.exceptions.ConnectionError:
        log.error("API unreachable. Is uvicorn running?")
        sys.exit(1)


def run_image_query(q: dict) -> dict:
    try:
        with open(q["image_path"], "rb") as f:
            img_bytes = f.read()

        t0 = time.perf_counter()
        resp = requests.post(
            f"{API_URL}/recommend/image",
            files={"file": (os.path.basename(q["image_path"]), img_bytes, "image/jpeg")},
            params={"top_k": TOP_K},
            timeout=30,
        )
        wall_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            log.warning(f"Image query failed [{resp.status_code}]: {q['product_id']}")
            return _failed_row("image", q["product_id"])

        results   = resp.json()
        infer_ms  = results[0]["inference_latency_ms"] if results else wall_ms
        cat_score = category_consistency(results, q["expected_category"])

        return {
            "type":                 "image",
            "query":                q["product_id"],
            "expected_category":    q["expected_category"],
            "precision_at_10":      cat_score,
            "category_consistency": cat_score,
            "latency_ms":           infer_ms,
            "result_count":         len(results),
        }
    except FileNotFoundError:
        log.warning(f"Image not found: {q['image_path']}")
        return _failed_row("image", q["product_id"])
    except requests.exceptions.ConnectionError:
        log.error("API unreachable. Is uvicorn running?")
        sys.exit(1)


def _failed_row(qtype: str, query: str) -> dict:
    return {
        "type":                 qtype,
        "query":                query,
        "expected_category":    "—",
        "precision_at_10":      0.0,
        "category_consistency": 0.0,
        "latency_ms":           -1.0,
        "result_count":         0,
    }


def print_table(rows: list[dict]) -> None:
    header = f"{'Type':<8}{'Query':<35}{'ExpCat':<18}{'P@10':>6}{'CatCon':>8}{'Lat(ms)':>10}"
    sep    = "─" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['type']:<8}"
            f"{str(r['query'])[:33]:<35}"
            f"{str(r['expected_category'])[:16]:<18}"
            f"{r['precision_at_10']:>6.2f}"
            f"{r['category_consistency']:>8.2f}"
            f"{r['latency_ms']:>10.1f}"
        )
    print(sep)

    text_rows  = [r for r in rows if r["type"] == "text"  and r["latency_ms"] > 0]
    image_rows = [r for r in rows if r["type"] == "image" and r["latency_ms"] > 0]
    all_valid  = [r for r in rows if r["latency_ms"] > 0]

    def avg(lst, key):
        return sum(x[key] for x in lst) / len(lst) if lst else 0.0

    print(f"\nSummary")
    print(f"  Text  queries  — Avg P@10: {avg(text_rows, 'precision_at_10'):.2f}  "
          f"Avg CatCon: {avg(text_rows, 'category_consistency'):.2f}  "
          f"Avg latency: {avg(text_rows, 'latency_ms'):.1f} ms")
    print(f"  Image queries  — Avg P@10: {avg(image_rows, 'precision_at_10'):.2f}  "
          f"Avg CatCon: {avg(image_rows, 'category_consistency'):.2f}  "
          f"Avg latency: {avg(image_rows, 'latency_ms'):.1f} ms")
    print(f"  Overall        — Avg P@10: {avg(all_valid, 'precision_at_10'):.2f}  "
          f"Avg CatCon: {avg(all_valid, 'category_consistency'):.2f}  "
          f"Avg latency: {avg(all_valid, 'latency_ms'):.1f} ms")


def main() -> None:
    log.info(f"Evaluating against {API_URL} …")

    # verify API reachable
    try:
        h = requests.get(f"{API_URL}/health", timeout=5)
        data = h.json()
        log.info(f"API healthy — {data.get('index_size', '?')} products indexed.")
    except Exception as exc:
        log.error(f"API health check failed: {exc}")
        sys.exit(1)

    df = load_metadata()
    image_queries = pick_image_queries(df)

    rows: list[dict] = []

    log.info("Running 10 text queries …")
    for q in TEXT_QUERIES:
        row = run_text_query(q)
        rows.append(row)
        log.info(f"  '{q['query'][:30]}' → P@10={row['precision_at_10']:.2f}  {row['latency_ms']:.1f}ms")

    log.info("Running 10 image queries …")
    for q in image_queries:
        row = run_image_query(q)
        rows.append(row)
        log.info(f"  img:{q['product_id']} → P@10={row['precision_at_10']:.2f}  {row['latency_ms']:.1f}ms")

    print_table(rows)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    log.info(f"Results saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
