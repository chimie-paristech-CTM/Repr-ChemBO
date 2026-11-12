"""Names and paths used by one Bayesian-optimisation run."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import traceback


CACHE_DIR = "./from_pretrained"
CACHE_ONLY = True

class ContextManager:
    """Manage run output and reusable precomputed-embedding paths.

    Directory layout::

        output/<run_id>/
            context.json
            results.pkl.gz

        pre_comp_emb/<dataset>/<fp>/<embedding_id>/
            features.pkl.gz
            context.json

    ``run_name`` is deterministic and depends only on ``config``. Information
    from the clock and Slurm is exposed separately, so the policy in
    :meth:`_get_run_id` can be changed without touching information collection.
    """

    OUTPUT_ROOT = Path("output")
    EMBEDDING_ROOT = Path("pre_comp_emb")
    FAILED_RUNS_ROOT = Path("failed_runs")
    HASH_LENGTH = 12

    def __init__(self, config: dict[str, Any]) -> None:
        # Use a monotonic clock for durations so wall-clock adjustments do not
        # affect the measured runtime.
        self._start_time = time.perf_counter()
        self._end_time: float | None = None

        self._validate_config(config)
        self.config = deepcopy(config)

        # Minimal information needed to identify and reproduce a run.
        self.timestamp = self.get_timestamp()
        self.random_id = self.get_random_id()
        self.slurm_info = self.get_slurm_info()

        self.run_name = self._get_run_name()
        self.run_id = self._get_run_id()

        self.output_root = self._resolve_root(self.OUTPUT_ROOT)
        self.embedding_root = self._resolve_root(self.EMBEDDING_ROOT)
        self.failed_runs_root = self._resolve_root(self.FAILED_RUNS_ROOT)

        self.output_dir = self._get_output_dir()
        self.run_config_path = self.output_dir / "context.json"
        self.result_path = self.output_dir / "results.pkl.gz"
        self.search_space_path = self.output_dir / "search_space.pkl.gz"
        self.failed_context_path = self.failed_runs_root / f"{self.run_id}.json"

        self.dataset_name = self._safe_name(self.config["dataset"]["name"])
        self.fp_name = self._safe_name(self.config["fp"]["name"])
        self.embedding_config = self._get_embedding_config()
        self.embedding_id = self._config_hash(self.embedding_config)
        self.embedding_dir = (
            self.embedding_root
            / self.dataset_name
            / self.fp_name
            / self.embedding_id
        )
        self.features_path = self.embedding_dir / "features.pkl.gz"
        self.embedding_config_path = self.embedding_dir / "context.json"
        
        self.campaign_name = self.run_name

    def get_elapsed_time(self) -> float:
        """Return elapsed wall time in seconds.

        Before :meth:`finish_timing` is called, this returns the current
        elapsed time. Afterwards, it returns the frozen total runtime.
        """
        end_time = self._end_time
        if end_time is None:
            end_time = time.perf_counter()
        return end_time - self._start_time

    def finish_timing(self) -> float:
        """Stop the timer and return the total runtime in seconds.

        This method is idempotent: repeated calls return the same value.
        """
        if self._end_time is None:
            self._end_time = time.perf_counter()
        return self.get_elapsed_time()

    def get_elapsed_time_hms(self) -> str:
        """Return the elapsed time formatted as ``HH:MM:SS``."""
        total_seconds = int(self.get_elapsed_time())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def save_failed_context(self, error: BaseException) -> Path:
        """Save this run's config to the shared failed-runs directory."""
        failure_dict = {
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        }
        return self.save_config(self.config|failure_dict, self.failed_context_path)
        

    @staticmethod
    def _resolve_root(path: str | Path) -> Path:
        return Path(path).expanduser().resolve(strict=False)

    @staticmethod
    def _validate_config(config: dict[str, Any]) -> None:
        """Validate the current expanded config schema used by this class."""
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")

        dataset = config.get("dataset")
        if not isinstance(dataset, dict):
            raise TypeError("config['dataset'] must be a dictionary")
        if not dataset.get("name"):
            raise ValueError("config['dataset']['name'] cannot be empty")
        if not isinstance(dataset.get("canonical_smiles"), bool):
            raise TypeError(
                "config['dataset']['canonical_smiles'] must be a boolean"
            )

        fp = config.get("fp")
        if not isinstance(fp, dict):
            raise TypeError("config['fp'] must be a dictionary")
        if not fp.get("name"):
            raise ValueError("config['fp']['name'] cannot be empty")
        if "class" in fp and not isinstance(fp["class"], str):
            raise TypeError("config['fp']['class'] must be a string")
        if not isinstance(fp.get("params"), dict):
            raise TypeError("config['fp']['params'] must be a dictionary")

    @staticmethod
    def canonical_config(config: dict[str, Any]) -> str:
        """Serialise the JSON config deterministically."""
        return json.dumps(
            config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @staticmethod
    def save_config(
        config: dict[str, Any],
        save_path: str | Path,
    ) -> Path:
        """Save a config(dict) as readable UTF-8 JSON and return its absolute path.

        The parent directory is created automatically.
        """
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")

        path = Path(save_path).expanduser().resolve(strict=False)
        path.parent.mkdir(parents=True, exist_ok=True)

        content = json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        path.write_text(f"{content}\n", encoding="utf-8")

        return path

    @classmethod
    def _config_hash(cls, config: dict[str, Any]) -> str:
        content = cls.canonical_config(config).encode("utf-8")
        return hashlib.sha256(content).hexdigest()[: cls.HASH_LENGTH]

    @staticmethod
    def _safe_name(value: Any) -> str:
        """Make a config value safe to use as one directory component."""
        name = str(value).strip()
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
        name = re.sub(r"[-_.]{2,}", "-", name).strip("-_.")
        if not name:
            raise ValueError(f"Cannot create a directory name from {value!r}")
        return name

    def _get_run_name(self) -> str:
        """Return a deterministic name based exclusively on the full config."""
        # This intentionally simple implementation can later be replaced with
        # a more readable config-to-name rule without changing the path layout.
        return self._config_hash(self.config)

    def _get_embedding_config(self) -> dict[str, Any]:
        """Return only the context needed to reproduce raw embeddings."""
        return {
            "dataset": self.config["dataset"]["name"],
            "fp": deepcopy(self.config["fp"]),
            "canonical_smiles": self.config["dataset"]["canonical_smiles"],
        }

    @staticmethod
    def get_timestamp() -> str:
        """Return the current UTC time in a filename-safe format."""
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    @staticmethod
    def get_random_id() -> str:
        """Return a random value for resolving otherwise identical runs."""
        return uuid.uuid4().hex

    @staticmethod
    def get_slurm_info() -> dict[str, str | None]:
        """Read the Slurm fields relevant to identifying one submitted task."""
        return {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "restart_count": os.environ.get("SLURM_RESTART_COUNT"),
        }

    def _get_run_id(self) -> str:
        """Combine the independent fields above into one execution ID.

        This method is deliberately kept small so the naming policy can be
        edited directly. ``random_id`` guarantees uniqueness, while the other
        fields make the ID traceable by a human.
        """
        job_id = self.slurm_info["job_id"]
        if job_id:
            restart = self.slurm_info["restart_count"]
            restart_part = f"-r{restart}" if restart not in (None, "", "0") else ""
            return self._safe_name(f"{self.run_name}-slurm-{job_id}{restart_part}")
        return self._safe_name(
            f"{self.run_name}-{self.timestamp}-{self.random_id[:8]}"
        )

    def _get_output_dir(self) -> Path:
        return self.output_root / self.run_id

    def prepare_output_dir(self) -> Path:
        """Create this run's private output directory.

        ``exist_ok=False`` detects an accidental identifier collision instead
        of silently allowing two processes to share a writable directory.
        """
        self.output_dir.mkdir(parents=True, exist_ok=False)
        return self.output_dir

    def prepare_embedding_dir(self) -> Path:
        """Create the directory for this exact embedding configuration."""
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        return self.embedding_dir

    def save_embedding_context(self) -> Path:
        """Save the context required to reproduce the cached embeddings."""
        return self.save_config(
            self.embedding_config,
            self.embedding_config_path,
        )

    def has_precomputed_embeddings(self) -> bool:
        """Return whether both cached features and their context are present."""
        return (
            self.features_path.is_file()
            and self.embedding_config_path.is_file()
        )

    def get_embedding_path(self, *, require_exists: bool = False) -> Path:
        """Return the path used to save or load precomputed features."""
        if require_exists and not self.has_precomputed_embeddings():
            raise FileNotFoundError(
                "No complete precomputed embedding cache for this context: "
                f"{self.embedding_dir}"
            )
        return self.features_path

    def metadata(self) -> dict[str, Any]:
        """Return the central names and paths for logging or inspection."""
        return {
            "run_name": self.run_name,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "slurm_info": dict(self.slurm_info),
            "output_dir": str(self.output_dir),
            "run_config_path": str(self.run_config_path),
            "result_path": str(self.result_path),
            "embedding_dir": str(self.embedding_dir),
            "embedding_id": self.embedding_id,
            "features_path": str(self.features_path),
            "embedding_config_path": str(self.embedding_config_path),
        }
