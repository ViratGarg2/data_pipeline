"""
Stream GPT-2 token shards from a Hugging Face dataset instead of downloading
the whole training set first.

Shards (`tokens/train-XXXXX.bin`, uint16) are fetched in a seeded random order
by a background thread and kept in a small on-disk pool. Batches are random
context-length windows, like `data.get_batch`, drawn from the pool; each shard
gets a budget of len(shard) / context_length windows (about one pass over its
tokens) and is deleted once the budget is spent. Disk use stays around
`pool_size + prefetch` shards (~100 MB each) instead of the full dataset.
"""

from __future__ import annotations

import logging
import os
import queue
import random
import threading
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class StreamingTokenDataset:
    def __init__(
        self,
        repo_id: str,
        cache_dir: str | Path,
        context_length: int,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        pool_size: int = 8,
        prefetch: int = 4,
        prefix: str = "tokens/",
    ):
        # Xet uploads/downloads hung on some networks; the plain LFS path is reliable.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.context_length = context_length
        self.pool_size = pool_size
        self.rng = random.Random(seed + rank)

        files = sorted(
            f for f in HfApi().list_repo_files(repo_id, repo_type="dataset")
            if f.startswith(prefix) and f.endswith(".bin")
        )
        if not files:
            raise FileNotFoundError(f"No {prefix}*.bin files in {repo_id}")
        # Same shuffled order for every run with this seed; each DDP rank takes a disjoint slice.
        random.Random(seed).shuffle(files)
        self.files = files[rank::world_size]
        logger.info(f"Streaming {len(self.files)} shards from {repo_id} (rank {rank}/{world_size})")

        self.ready: queue.Queue = queue.Queue(maxsize=prefetch)
        self.pool: list[list] = []  # [path, memmap, windows_left]
        self.shards_consumed = 0
        self.epoch = 0
        self._thread = threading.Thread(target=self._download_loop, daemon=True)
        self._thread.start()

    def _download_loop(self):
        from huggingface_hub import hf_hub_download

        while True:
            for name in self.files:
                for attempt in range(10):
                    try:
                        path = hf_hub_download(
                            self.repo_id, name, repo_type="dataset", local_dir=self.cache_dir,
                        )
                        break
                    except Exception as e:  # network hiccup: retry with backoff
                        logger.warning(f"download of {name} failed ({e}); retry {attempt + 1}")
                        threading.Event().wait(min(60, 5 * 2 ** attempt))
                else:
                    raise RuntimeError(f"could not download {name}")
                self.ready.put(path)
            # Only reached if training needs more than one pass over the data.
            self.epoch += 1
            logger.warning(f"all shards streamed; starting pass {self.epoch + 1}")

    def _fill_pool(self):
        while len(self.pool) < self.pool_size:
            block = not self.pool  # only wait when there is nothing to sample from
            try:
                path = self.ready.get(block=block, timeout=None if block else 0)
            except queue.Empty:
                return
            data = np.memmap(path, dtype=np.uint16, mode="r")
            self.pool.append([path, data, max(1, len(data) // self.context_length)])

    def _retire(self, idx: int):
        path, data, _ = self.pool.pop(idx)
        del data
        Path(path).unlink(missing_ok=True)
        self.shards_consumed += 1

    def get_batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        self._fill_pool()
        xs, ys = [], []
        for _ in range(batch_size):
            # Pick a shard in proportion to its remaining budget, then a random window in it.
            weights = [entry[2] for entry in self.pool]
            idx = self.rng.choices(range(len(self.pool)), weights=weights)[0]
            data = self.pool[idx][1]
            start = self.rng.randrange(len(data) - self.context_length)
            chunk = torch.from_numpy(data[start:start + self.context_length + 1].astype(np.int64))
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            self.pool[idx][2] -= 1
            if self.pool[idx][2] <= 0:
                self._retire(idx)
                self._fill_pool()
        x, y = torch.stack(xs), torch.stack(ys)
        if "cuda" in device:
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)
