"""
MoE Expert Sniper — Read only active experts from SSD via F_NOCACHE + pread.

For a 256-expert model with 8 active per token:
  - Each expert: ~1.8 MB (4-bit quantized)
  - Per layer: 8 × 1.8 MB = 14.4 MB
  - Per token (40 layers): 576 MB
  - At 5 GB/s NVMe: 115ms = 8.7 tok/s theoretical

Uses multi-threaded pread (8 workers) to saturate NVMe queue depth.
"""

import os
import json
import fcntl
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

F_NOCACHE = 48
PAGE_SIZE = 16384


class MoEExpertReader:
    """
    Reads specific experts from concatenated layer files via F_NOCACHE + pread.
    Expert offset = data_start + expert_id × expert_block_size
    """

    def __init__(self, expert_dir, num_layers, num_workers=8):
        self.expert_dir = expert_dir
        self.num_layers = num_layers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # Parse all layer headers
        self.headers = {}
        self.fds = {}
        for i in range(num_layers):
            path = f"{expert_dir}/layer_{i:02d}.bin"
            with open(path, "rb") as f:
                raw = f.read(PAGE_SIZE)
            self.headers[i] = json.loads(raw.rstrip(b"\x00"))

        # Precompute layout info
        h0 = self.headers[0]["layout"]
        self.expert_block_size = h0["expert_block_size"]
        self.data_start = h0["data_start"]
        self.tensor_layout = h0["tensors"]

        # Stats
        self.read_time = 0.0
        self.reads = 0
        self.bytes_read = 0

        # Prefetch state
        self.prefetch_futures = {}

    def _get_fd(self, layer_idx):
        if layer_idx not in self.fds:
            path = f"{self.expert_dir}/layer_{layer_idx:02d}.bin"
            fd = os.open(path, os.O_RDONLY)
            fcntl.fcntl(fd, F_NOCACHE, 1)
            self.fds[layer_idx] = fd
        return self.fds[layer_idx]

    def _read_expert(self, layer_idx, expert_id):
        """Read one expert's data via pread. Thread-safe."""
        fd = self._get_fd(layer_idx)
        offset = self.data_start + expert_id * self.expert_block_size

        # Read the full expert block
        data = os.pread(fd, self.expert_block_size, offset)
        return data

    def _parse_expert_data(self, raw_data, expert_id):
        """Parse raw bytes into MLX arrays for one expert."""
        import mlx.core as mx

        # Map dtype strings to MLX dtypes
        MLX_DTYPES = {
            "uint32": mx.uint32, "float16": mx.float16, "float32": mx.float32,
            "bfloat16": mx.bfloat16,
        }

        result = {}
        for name, info in self.tensor_layout.items():
            inner_offset = info["inner_offset"]
            nbytes = info["nbytes"]
            shape = info["shape_per_expert"]
            dtype_str = info["dtype"].replace("mlx.core.", "")
            mlx_dtype = MLX_DTYPES.get(dtype_str, mx.float16)

            arr_bytes = raw_data[inner_offset:inner_offset + nbytes]
            # Create MLX array directly from bytes (handles bfloat16 correctly)
            flat = mx.array(np.frombuffer(arr_bytes, dtype=np.uint8))
            arr = flat.view(mlx_dtype).reshape(shape)
            result[name] = arr

        return result

    def prefetch_experts(self, layer_idx, expert_ids):
        """Launch parallel pread for all active experts. Non-blocking."""
        futures = {}
        for eid in expert_ids:
            future = self.executor.submit(self._read_expert, layer_idx, eid)
            futures[eid] = future
        self.prefetch_futures[layer_idx] = futures

    def get_experts(self, layer_idx, expert_ids):
        """
        Get parsed expert data for active experts.
        Uses prefetched data if available, otherwise reads synchronously.

        Returns: dict[expert_id] → dict[tensor_name → mx.array]
        """
        t0 = time.time()

        experts = {}
        if layer_idx in self.prefetch_futures:
            # Use prefetched data
            futures = self.prefetch_futures.pop(layer_idx)
            for eid in expert_ids:
                if eid in futures:
                    raw = futures[eid].result()
                else:
                    raw = self._read_expert(layer_idx, eid)
                experts[eid] = self._parse_expert_data(raw, eid)
                self.bytes_read += len(raw)
        else:
            # Synchronous read
            for eid in expert_ids:
                raw = self._read_expert(layer_idx, eid)
                experts[eid] = self._parse_expert_data(raw, eid)
                self.bytes_read += len(raw)

        self.read_time += time.time() - t0
        self.reads += len(expert_ids)
        return experts

    def stats(self):
        if self.reads == 0:
            return "No reads yet"
        avg_ms = self.read_time / self.reads * 1000
        throughput = self.bytes_read / self.read_time / 1e9 if self.read_time > 0 else 0
        return (f"reads={self.reads}, avg={avg_ms:.1f}ms/expert, "
                f"throughput={throughput:.1f} GB/s, "
                f"total_bytes={self.bytes_read/1e9:.2f} GB, "
                f"total_time={self.read_time:.1f}s")

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.executor.shutdown(wait=False)
