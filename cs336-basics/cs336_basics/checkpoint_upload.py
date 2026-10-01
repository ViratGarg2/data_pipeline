"""
Push training checkpoints to a Hugging Face model repo without pausing training.

Uploads run one at a time on a background thread. Each checkpoint goes to its own
folder in the repo (step_0000002000/, step_0000004000/, ...), so every checkpoint
stays available on the Hub even after the local copy is deleted; the final model is
uploaded to the repo root. Folders stay "pending" until their upload finishes, so
callers can avoid deleting a checkpoint that is still being sent.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointUploader:
    def __init__(self, repo_id: str, private: bool = True, retries: int = 5):
        # Xet uploads hung on some networks; the plain LFS path is reliable.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from huggingface_hub import HfApi

        self.api = HfApi()
        self.repo_id = repo_id
        self.retries = retries
        self.api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        self._queue: queue.Queue = queue.Queue()
        self._pending: set[Path] = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Pushing checkpoints to https://huggingface.co/{repo_id}")

    def submit(
        self, folder: Path, message: str, path_in_repo: str = "", allow_patterns: list[str] | None = None
    ) -> None:
        folder = Path(folder)
        with self._lock:
            self._pending.add(folder)
        self._queue.put((folder, message, path_in_repo, allow_patterns))

    def is_pending(self, folder: Path) -> bool:
        with self._lock:
            return Path(folder) in self._pending

    def _run(self) -> None:
        while True:
            folder, message, path_in_repo, allow_patterns = self._queue.get()
            if folder is None:
                return
            start = time.time()
            for attempt in range(1, self.retries + 1):
                try:
                    self.api.upload_folder(
                        repo_id=self.repo_id,
                        folder_path=str(folder),
                        path_in_repo=path_in_repo,
                        allow_patterns=allow_patterns,
                        commit_message=message,
                    )
                    logger.info(f"Uploaded {folder.name} to {self.repo_id} in {time.time() - start:.0f}s")
                    break
                except Exception as e:  # network trouble must never stop training
                    logger.warning(f"Upload of {folder.name} failed ({e}); attempt {attempt}/{self.retries}")
                    time.sleep(60 * attempt)
            with self._lock:
                self._pending.discard(folder)

    def close(self, timeout: float = 1800) -> None:
        """Wait for queued uploads to finish, up to ``timeout`` seconds."""
        self._queue.put((None, None, None, None))
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.warning(f"Checkpoint uploads still running after {timeout:.0f}s; exiting anyway")
