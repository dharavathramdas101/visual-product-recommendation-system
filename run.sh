#!/bin/bash
set -e

PYTHON="python"
ROOT="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT"

echo "============================================================"
echo " Visual Product Recommendation System"
echo "============================================================"

echo ""
echo "Step 1: Generating embeddings (CLIP ViT-B/32, up to 5000 images)..."
"$PYTHON" models/embeddings.py

echo ""
echo "Step 2: Building FAISS index..."
"$PYTHON" models/build_index.py

echo ""
echo "Step 3: Starting FastAPI backend on http://0.0.0.0:8000 ..."
"$PYTHON" -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!
echo "  FastAPI PID: $API_PID"

# wait for API to be ready
echo "  Waiting for API to start..."
for i in {1..20}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "  API is ready."
        break
    fi
    sleep 2
done

echo ""
echo "Step 4: Starting Streamlit frontend on http://localhost:8501 ..."
"$PYTHON" -m streamlit run app/streamlit_app.py --server.port 8501 &
ST_PID=$!
echo "  Streamlit PID: $ST_PID"

echo ""
echo "============================================================"
echo " Services running:"
echo "   FastAPI  → http://localhost:8000"
echo "   Docs     → http://localhost:8000/docs"
echo "   Streamlit→ http://localhost:8501"
echo " Press Ctrl+C to stop all services."
echo "============================================================"

# wait for Ctrl+C then kill both
trap "echo 'Stopping...'; kill $API_PID $ST_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait $API_PID $ST_PID
