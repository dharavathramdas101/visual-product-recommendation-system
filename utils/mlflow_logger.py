"""
MLflow logger for the Visual Product Recommender.

One persistent run is created at startup; all per-request metrics are
logged as steps on that run. Thread-safe step counter.
"""

import threading
import logging
import mlflow

log = logging.getLogger(__name__)

EXPERIMENT_NAME = "visual-recommender"
TRACKING_URI    = "sqlite:///mlflow.db"   # MLflow 3.x requires DB backend


class MLflowLogger:
    def __init__(self) -> None:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        self._run = mlflow.start_run(run_name="api-session")
        self._run_id = self._run.info.run_id
        self._step = 0
        self._lock = threading.Lock()

        log.info(f"MLflow run started: {self._run_id}  (experiment={EXPERIMENT_NAME})")

    # ── startup ────────────────────────────────────────────────────────────────
    def log_startup(
        self,
        *,
        dataset_size: int,
        model_name: str,
        index_type: str,
        embedding_dim: int,
    ) -> None:
        mlflow.log_params(
            {
                "model_name":    model_name,
                "index_type":    index_type,
                "embedding_dim": embedding_dim,
                "dataset_size":  dataset_size,
            }
        )
        log.info(
            f"MLflow startup params logged: model={model_name} "
            f"index={index_type} dim={embedding_dim} dataset={dataset_size}"
        )

    # ── per-request ────────────────────────────────────────────────────────────
    def log_request(
        self,
        *,
        query_type: str,
        top_k: int,
        latency_ms: float,
        result_count: int,
    ) -> None:
        with self._lock:
            step = self._step
            self._step += 1

        try:
            mlflow.log_metrics(
                {
                    f"{query_type}_latency_ms": latency_ms,
                    f"{query_type}_result_count": float(result_count),
                    f"{query_type}_top_k": float(top_k),
                },
                step=step,
            )
        except Exception as exc:
            log.warning(f"MLflow log_request failed (non-fatal): {exc}")

    # ── cleanup ────────────────────────────────────────────────────────────────
    def end_run(self) -> None:
        try:
            mlflow.end_run()
            log.info("MLflow run ended.")
        except Exception:
            pass
