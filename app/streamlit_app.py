"""
Streamlit frontend — Visual Product Recommender (Premium UI)
"""

import io
import time
import requests
import streamlit as st
from PIL import Image

# ── constants ───────────────────────────────────────────────────────────────────
DEFAULT_API = "http://localhost:8000"
COLS        = 5

# initialise session state before any widget renders
if "api_url" not in st.session_state:
    st.session_state["api_url"] = DEFAULT_API

# ── page config (MUST be first) ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Visual Product Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── premium CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 2rem 3rem; max-width: 1400px; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(255,215,0,0.15), transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0;
    background: linear-gradient(90deg, #ffffff, #f0c040);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.6);
    margin-top: 0.4rem;
    font-weight: 400;
    letter-spacing: 0.5px;
}
.hero-badges {
    display: flex;
    gap: 0.6rem;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.badge {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 500;
    backdrop-filter: blur(4px);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.05);
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stTextInput > label { color: rgba(255,255,255,0.6) !important; font-size:0.8rem !important; }
[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: white !important;
    border-radius: 8px !important;
}
.sidebar-logo {
    font-size: 1.2rem;
    font-weight: 700;
    color: white;
    padding: 1rem 0 0.5rem 0;
    letter-spacing: -0.3px;
}
.sidebar-section {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: rgba(255,255,255,0.35) !important;
    margin: 1.2rem 0 0.5rem 0;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    gap: 0.3rem;
    border-bottom: 2px solid #f0f0f0;
    padding-bottom: 0;
}
[data-testid="stTabs"] [role="tab"] {
    background: transparent;
    border: none;
    border-radius: 8px 8px 0 0;
    padding: 0.7rem 1.4rem;
    font-weight: 500;
    font-size: 0.9rem;
    color: #666;
    transition: all 0.2s;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #302b63, #24243e);
    color: white !important;
    box-shadow: 0 -2px 12px rgba(48,43,99,0.3);
}

/* ── Product card ── */
.product-card {
    background: white;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    border: 1px solid rgba(0,0,0,0.05);
    height: 100%;
}
.product-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.15);
}
.card-img-wrap {
    background: #f8f8f8;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 180px;
    overflow: hidden;
}
.card-img-wrap img {
    max-height: 170px;
    max-width: 100%;
    object-fit: contain;
    padding: 8px;
}
.card-body {
    padding: 0.75rem 0.85rem 0.85rem;
}
.card-name {
    font-size: 0.78rem;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 0.35rem;
    line-height: 1.3;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.card-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.4rem;
}
.card-cat {
    background: #f0edff;
    color: #302b63;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.2rem 0.5rem;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.card-score {
    font-size: 0.75rem;
    font-weight: 700;
    color: #f0a500;
}

/* ── Score bar ── */
.score-bar-wrap {
    margin-top: 0.4rem;
    background: #f0f0f0;
    border-radius: 999px;
    height: 3px;
    overflow: hidden;
}
.score-bar {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #302b63, #f0c040);
}

/* ── Metrics row ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.metric-box {
    background: white;
    border: 1px solid #eee;
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    min-width: 160px;
}
.metric-icon { font-size: 1.4rem; }
.metric-label { font-size: 0.7rem; color: #999; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-size: 1.1rem; font-weight: 700; color: #1a1a2e; }

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #d0cdff !important;
    border-radius: 16px !important;
    background: #faf9ff !important;
    padding: 0.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #302b63 !important;
}

/* ── Search button ── */
.stButton > button {
    background: linear-gradient(135deg, #302b63, #0f0c29) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 14px rgba(48,43,99,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(48,43,99,0.45) !important;
}

/* ── Text input ── */
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 2px solid #e8e6ff !important;
    padding: 0.6rem 1rem !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #302b63 !important;
    box-shadow: 0 0 0 3px rgba(48,43,99,0.1) !important;
}

/* ── Alert boxes ── */
.stSuccess {
    border-radius: 12px !important;
    border-left: 4px solid #00c49f !important;
}
.stWarning {
    border-radius: 12px !important;
    border-left: 4px solid #f0a500 !important;
}
.stError {
    border-radius: 12px !important;
    border-left: 4px solid #ff4b6e !important;
}

/* ── Section title ── */
.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── No results ── */
.no-results {
    text-align: center;
    padding: 3rem;
    color: #999;
    font-size: 0.95rem;
}

/* ── Divider ── */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #e0e0e0, transparent);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── helpers ──────────────────────────────────────────────────────────────────────
def get_api_url() -> str:
    val = st.session_state.get("api_url", "").strip()
    return (val or DEFAULT_API).rstrip("/")


def fetch_image(api_url: str, image_path: str, upscale_to: int = 300) -> Image.Image | None:
    # try primary path, then myntradataset fallback
    paths = [image_path, image_path.replace("images/", "myntradataset/images/", 1)]
    for path in paths:
        try:
            resp = requests.get(f"{api_url}/{path}", timeout=5)
            if resp.status_code == 200:
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                w, h = img.size
                if w < upscale_to or h < upscale_to:
                    scale = upscale_to / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                return img
        except Exception:
            continue
    return None


def score_color(score: float) -> str:
    if score >= 0.90: return "#00c49f"
    if score >= 0.75: return "#f0a500"
    return "#ff6b6b"


def render_product_card(product: dict, api_url: str) -> None:
    pid    = product.get("product_id", "—")
    name   = product.get("product_name") or product.get("productDisplayName") or pid
    cat    = product.get("category", "—")
    g      = product.get("gender", "")
    score  = product.get("similarity_score", 0.0)
    path   = product.get("image_path", "")

    img = fetch_image(api_url, path)
    img_b64 = ""
    if img:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        import base64
        img_b64 = base64.b64encode(buf.getvalue()).decode()

    img_html = (
        f'<img src="data:image/jpeg;base64,{img_b64}" />'
        if img_b64
        else '<div style="height:170px;display:flex;align-items:center;justify-content:center;color:#ccc;font-size:2rem;">📷</div>'
    )

    bar_width = int(score * 100)
    short_name = (name[:28] + "…") if len(name) > 28 else name

    st.markdown(f"""
    <div class="product-card">
        <div class="card-img-wrap">{img_html}</div>
        <div class="card-body">
            <div class="card-name" title="{name}">{short_name}</div>
            <div class="card-meta">
                <span class="card-cat">{cat}</span>
                {f'<span class="card-cat" style="background:#e8f4ff;color:#0066cc">{g}</span>' if g else ''}
                <span class="card-score">{score:.3f}</span>
            </div>
            <div class="score-bar-wrap">
                <div class="score-bar" style="width:{bar_width}%"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(n_results: int, infer_ms: float, wall_ms: float) -> None:
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <span class="metric-icon">🎯</span>
            <div>
                <div class="metric-label">Results</div>
                <div class="metric-value">{n_results}</div>
            </div>
        </div>
        <div class="metric-box">
            <span class="metric-icon">⚡</span>
            <div>
                <div class="metric-label">Inference</div>
                <div class="metric-value">{infer_ms:.1f} ms</div>
            </div>
        </div>
        <div class="metric-box">
            <span class="metric-icon">🌐</span>
            <div>
                <div class="metric-label">Wall time</div>
                <div class="metric-value">{wall_ms:.0f} ms</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_results(results: list[dict], api_url: str, infer_ms: float, wall_ms: float) -> None:
    if not results:
        st.markdown('<div class="no-results">No results found.</div>', unsafe_allow_html=True)
        return

    render_metrics(len(results), infer_ms, wall_ms)
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    cols = st.columns(COLS, gap="medium")
    for i, product in enumerate(results):
        with cols[i % COLS]:
            render_product_card(product, api_url)


# ── hero header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🛍️ Visual Product Recommender</div>
    <div class="hero-sub">Cross-modal fashion search powered by CLIP ViT-B/32 &amp; FAISS</div>
    <div class="hero-badges">
        <span class="badge">CLIP ViT-B/32</span>
        <span class="badge">FAISS IndexFlatIP</span>
        <span class="badge">30,000 Products</span>
        <span class="badge">512-dim Embeddings</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚙️ Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">API Configuration</div>', unsafe_allow_html=True)
    api_url = st.text_input("Endpoint", placeholder=DEFAULT_API, key="api_url", label_visibility="collapsed")

    st.markdown('<div class="sidebar-section">Search Parameters</div>', unsafe_allow_html=True)
    top_k  = st.slider("Top K results", min_value=5, max_value=20, value=10, step=1)
    gender = st.radio("Gender filter", ["All", "Men", "Women", "Boys", "Girls", "Unisex"],
                      horizontal=False, key="gender_filter")

    st.markdown('<div class="sidebar-section">Status</div>', unsafe_allow_html=True)
    if st.button("🔍 Health Check", width='stretch'):
        try:
            r = requests.get(f"{get_api_url()}/health", timeout=5)
            d = r.json()
            st.success(f"✅ Online · {d.get('index_size','?')} products")
            st.caption(f"Model: {d.get('device','cpu').upper()}")
        except Exception as exc:
            st.error(f"❌ Unreachable\n{exc}")

    st.markdown("---")
    st.markdown("""
    <div style="color:rgba(255,255,255,0.3);font-size:0.7rem;line-height:1.6">
    Image Search — upload product photo<br>
    Text Search — describe in natural language<br>
    Hybrid — combine both for best results
    </div>
    """, unsafe_allow_html=True)

# ── tabs ─────────────────────────────────────────────────────────────────────────
tab_image, tab_text, tab_hybrid = st.tabs(
    ["🖼️  Image Search", "🔤  Text Search", "⚡  Hybrid Search"]
)

# ── Image Search ─────────────────────────────────────────────────────────────────
with tab_image:
    st.markdown('<div class="section-title">🖼️ Search by Image</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        uploaded = st.file_uploader(
            "Drop or click to upload", type=["jpg","jpeg","png","webp"], key="img_upload",
            label_visibility="collapsed",
        )
        if uploaded:
            st.image(uploaded, caption="Query image", width='stretch')
            search_btn = st.button("🔍 Find Similar Products", key="btn_img", width='stretch')
        else:
            st.markdown("""
            <div style="text-align:center;padding:2rem;color:#aaa;border:2px dashed #d0cdff;
                        border-radius:16px;background:#faf9ff">
                <div style="font-size:2.5rem">📸</div>
                <div style="font-size:0.85rem;margin-top:0.5rem">Upload a product image to begin</div>
            </div>""", unsafe_allow_html=True)
            search_btn = False

    with c2:
        if uploaded and search_btn:
            with st.spinner("Encoding image and searching …"):
                try:
                    t0 = time.perf_counter()
                    resp = requests.post(
                        f"{get_api_url()}/recommend/image",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        params={"top_k": top_k, "gender": gender},
                        timeout=30,
                    )
                    wall_ms = (time.perf_counter() - t0) * 1000

                    if resp.status_code == 200:
                        results = resp.json()
                        infer_ms = results[0]["inference_latency_ms"] if results else wall_ms
                        render_results(results, get_api_url(), infer_ms, wall_ms)
                    else:
                        st.error(f"API error {resp.status_code}: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Is uvicorn running on port 8000?")
        elif not uploaded:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;
                        height:300px;color:#ccc;flex-direction:column;gap:1rem">
                <div style="font-size:3rem">👈</div>
                <div style="font-size:0.9rem">Upload an image to see recommendations</div>
            </div>""", unsafe_allow_html=True)

# ── Text Search ───────────────────────────────────────────────────────────────────
with tab_text:
    st.markdown('<div class="section-title">🔤 Search by Text</div>', unsafe_allow_html=True)

    t_col1, t_col2 = st.columns([1, 3], gap="large")
    with t_col1:
        suggestions = ["blue denim jeans", "white sneakers", "black leather bag", "floral kurta", "sports shorts"]

        def _set_suggestion(s: str) -> None:
            st.session_state["txt_query"] = s

        st.markdown("""
        <div style="font-size:0.72rem;color:#999;margin-bottom:0.4rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">Try these</div>
        """, unsafe_allow_html=True)
        for s in suggestions:
            st.button(s, key=f"sug_{s}", width='stretch', on_click=_set_suggestion, args=(s,))

        text_query = st.text_input(
            "Describe the product",
            placeholder="e.g. red summer dress",
            key="txt_query",
        )

        txt_search = st.button("🔍 Search", key="btn_txt", width='stretch')

    with t_col2:
        if txt_search and text_query and text_query.strip():
            with st.spinner(f'Searching for "{text_query}" …'):
                try:
                    t0 = time.perf_counter()
                    resp = requests.post(
                        f"{get_api_url()}/recommend/text",
                        data={"query": text_query.strip(), "top_k": top_k, "gender": gender},
                        timeout=30,
                    )
                    wall_ms = (time.perf_counter() - t0) * 1000

                    if resp.status_code == 200:
                        results = resp.json()
                        infer_ms = results[0]["inference_latency_ms"] if results else wall_ms
                        render_results(results, get_api_url(), infer_ms, wall_ms)
                    else:
                        st.error(f"API error {resp.status_code}: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Is uvicorn running on port 8000?")
        elif txt_search and not text_query.strip():
            st.warning("Enter a search query first.")
        else:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;
                        height:300px;color:#ccc;flex-direction:column;gap:1rem">
                <div style="font-size:3rem">💬</div>
                <div style="font-size:0.9rem">Type a description to find matching products</div>
            </div>""", unsafe_allow_html=True)

# ── Hybrid Search ─────────────────────────────────────────────────────────────────
with tab_hybrid:
    alpha = st.slider(
        "🔤 Text  ◀────────────────────▶  Image 🖼️",
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        key="alpha_slider",
        help="Slide LEFT for more text influence · Slide RIGHT for more image influence",
    )
    st.markdown(
        f'<div class="section-title">⚡ Hybrid Search '
        f'<span style="font-size:0.75rem;font-weight:400;color:#999;margin-left:0.5rem">'
        f'🖼️ Image {int(alpha*100)}% · 🔤 Text {int((1-alpha)*100)}%</span></div>',
        unsafe_allow_html=True,
    )

    h1, h2 = st.columns([1, 2], gap="large")
    with h1:
        h_uploaded = st.file_uploader(
            "Upload reference image",
            type=["jpg","jpeg","png","webp"],
            key="hybrid_upload",
            label_visibility="collapsed",
        )
        if h_uploaded:
            st.image(h_uploaded, caption="Reference image", width='stretch')

        h_query = st.text_input(
            "Refine with text",
            placeholder="e.g. same style but in blue",
            key="hybrid_query",
        )

        hybrid_btn = st.button("⚡ Hybrid Search", key="btn_hybrid", width='stretch')

        if not h_uploaded:
            st.markdown("""
            <div style="margin-top:0.5rem;font-size:0.75rem;color:#aaa;text-align:center">
            Upload an image + describe what you want.<br>CLIP fuses both signals for precise results.
            </div>""", unsafe_allow_html=True)

    with h2:
        if hybrid_btn:
            if not h_uploaded:
                st.warning("Please upload a reference image.")
            elif not h_query.strip():
                st.warning("Please enter a text description.")
            else:
                with st.spinner("Fusing image + text signals …"):
                    try:
                        t0 = time.perf_counter()
                        resp = requests.post(
                            f"{get_api_url()}/recommend/hybrid",
                            files={"file": (h_uploaded.name, h_uploaded.getvalue(), h_uploaded.type)},
                            data={"query": h_query.strip(), "top_k": top_k, "alpha": alpha, "gender": gender},
                            timeout=30,
                        )
                        wall_ms = (time.perf_counter() - t0) * 1000

                        if resp.status_code == 200:
                            results = resp.json()
                            infer_ms = results[0]["inference_latency_ms"] if results else wall_ms
                            render_results(results, get_api_url(), infer_ms, wall_ms)
                        else:
                            st.error(f"API error {resp.status_code}: {resp.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to API. Is uvicorn running on port 8000?")
        else:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;
                        height:350px;color:#ccc;flex-direction:column;gap:1rem">
                <div style="font-size:3rem">🔮</div>
                <div style="font-size:0.9rem;text-align:center;max-width:240px">
                    Combine an image + text for the most precise cross-modal search
                </div>
            </div>""", unsafe_allow_html=True)
