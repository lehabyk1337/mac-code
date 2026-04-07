#!/usr/bin/env python3
"""
mac-tensor bench — Throughput benchmark for distributed inference.

Hits a running mac-tensor UI server with N concurrent clients, each sending
M requests. Measures aggregate tokens/sec across all clients.

This is the throughput angle: distributed M2 cluster at $0.40/hr can beat
a single Mac on $/token when serving multiple concurrent users, even though
single-stream latency is worse.

Usage:
    mac-tensor bench --server http://localhost:8500 --concurrent 4 --requests 5

Output:
    [c0 r0] 50 tokens in 38.5s = 1.30 tok/s
    [c1 r0] 52 tokens in 41.2s = 1.26 tok/s
    ...
    Aggregate: 4 clients × 5 requests = 20 requests
    Total tokens: 1024
    Wall time: 158.3s
    Aggregate throughput: 6.47 tok/s
    Per-client: 1.62 tok/s avg
    $/1M tokens (at $0.40/hr): $0.017
"""

import argparse
import json
import sys
import time
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed


PROMPTS = [
    "What is the capital of France?",
    "Explain quantum entanglement in one sentence.",
    "Write a haiku about Apple Silicon.",
    "What is 17 * 23?",
    "Name three benefits of MoE architectures.",
    "What's the difference between recall and precision?",
    "Define gradient descent in 20 words.",
    "Why is unified memory good for ML inference?",
    "What is the speed of light in m/s?",
    "List three popular MLX packages.",
]


def detect_endpoint(server_url):
    """Probe /api/info to figure out which endpoint to use."""
    try:
        with urllib.request.urlopen(f"{server_url}/api/info", timeout=5) as r:
            info = json.loads(r.read())
            if info.get("vision"):
                return "/api/chat_vision", "vision"
            else:
                return "/api/chat", "text"
    except Exception:
        return "/api/chat", "text"


def send_request(server_url, endpoint, mode, client_id, request_id, prompt, max_tokens):
    """Send one chat request and return (tokens_generated, elapsed_seconds)."""
    if mode == "vision":
        # Vision endpoint uses multipart/form-data; we send no image
        boundary = f"----maccastpacker{client_id}{request_id}{int(time.time()*1000)}"
        body_parts = []
        body_parts.append(f"--{boundary}\r\n")
        body_parts.append('Content-Disposition: form-data; name="message"\r\n\r\n')
        body_parts.append(prompt)
        body_parts.append(f"\r\n--{boundary}\r\n")
        body_parts.append('Content-Disposition: form-data; name="max_tokens"\r\n\r\n')
        body_parts.append(str(max_tokens))
        body_parts.append(f"\r\n--{boundary}--\r\n")
        body = "".join(body_parts).encode()
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    else:
        body = json.dumps({
            "message": prompt,
            "max_iterations": 1,
            "max_tokens": max_tokens,
        }).encode()
        headers = {"Content-Type": "application/json"}

    req = urllib.request.Request(
        f"{server_url}{endpoint}", data=body, headers=headers,
    )

    t_start = time.time()
    tokens_generated = 0

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            buffer = ""
            for chunk in iter(lambda: resp.read(1024), b""):
                buffer += chunk.decode("utf-8", errors="ignore")
                while "\n\n" in buffer:
                    event_str, buffer = buffer.split("\n\n", 1)
                    if event_str.startswith("data: "):
                        try:
                            event = json.loads(event_str[6:])
                            if event.get("type") == "token":
                                tokens_generated += 1
                            elif event.get("type") == "done":
                                break
                        except json.JSONDecodeError:
                            pass
    except Exception as e:
        elapsed = time.time() - t_start
        print(f"  [c{client_id} r{request_id}] ERROR after {elapsed:.1f}s: {e}")
        return 0, elapsed

    elapsed = time.time() - t_start
    return tokens_generated, elapsed


def run_client(server_url, endpoint, mode, client_id, num_requests, max_tokens):
    """Single client running num_requests sequentially."""
    results = []
    for r in range(num_requests):
        prompt = PROMPTS[(client_id * 7 + r) % len(PROMPTS)]
        tokens, elapsed = send_request(server_url, endpoint, mode, client_id, r, prompt, max_tokens)
        rate = tokens / elapsed if elapsed > 0 else 0
        print(f"  [c{client_id} r{r}] {tokens} tokens in {elapsed:.1f}s = {rate:.2f} tok/s",
              flush=True)
        results.append((tokens, elapsed))
    return results


def main(args):
    server = args.server.rstrip("/")
    n_clients = args.concurrent
    n_requests = args.requests
    max_tokens = args.max_tokens

    print("=" * 60)
    print(f"mac-tensor benchmark")
    print(f"  Server:     {server}")
    print(f"  Clients:    {n_clients}  (concurrent)")
    print(f"  Per-client: {n_requests} requests")
    print(f"  Total:      {n_clients * n_requests} requests")
    print(f"  Max tokens: {max_tokens}")
    print("=" * 60)

    # Verify server is reachable + auto-detect endpoint
    endpoint, mode = detect_endpoint(server)
    try:
        with urllib.request.urlopen(f"{server}/api/info", timeout=5) as r:
            info = json.loads(r.read())
            nodes = info.get('nodes')
            n_nodes = len(nodes) if nodes else 0
            backend_label = f"{info.get('model')} ({mode}"
            if mode == "vision":
                backend_label += ", single Mac"
            else:
                backend_label += f", {n_nodes} nodes"
            backend_label += ")"
            print(f"  Backend:    {backend_label}")
            print(f"  Endpoint:   {endpoint}")
    except Exception as e:
        print(f"Cannot reach server at {server}: {e}")
        sys.exit(1)

    print(f"\nStarting {n_clients} concurrent clients...\n")

    t_start = time.time()
    all_results = []

    with ThreadPoolExecutor(max_workers=n_clients) as pool:
        futures = [
            pool.submit(run_client, server, endpoint, mode, i, n_requests, max_tokens)
            for i in range(n_clients)
        ]
        for f in as_completed(futures):
            all_results.extend(f.result())

    wall = time.time() - t_start

    total_tokens = sum(t for t, _ in all_results)
    total_compute_time = sum(e for _, e in all_results)
    n_completed = len(all_results)
    aggregate_tps = total_tokens / wall if wall > 0 else 0
    per_client_tps = aggregate_tps / n_clients if n_clients > 0 else 0
    avg_per_request_tps = total_tokens / total_compute_time if total_compute_time > 0 else 0

    # Cost calc — assume $0.40/hr for the cluster
    HOURLY_COST = args.hourly_cost
    cost_for_run = (wall / 3600) * HOURLY_COST
    cost_per_million = (cost_for_run / total_tokens * 1_000_000) if total_tokens > 0 else 0

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Requests completed:    {n_completed}/{n_clients * n_requests}")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"  Wall clock time:        {wall:.1f}s")
    print(f"  Aggregate throughput:   {aggregate_tps:.2f} tok/s")
    print(f"  Per-client average:     {per_client_tps:.2f} tok/s")
    print(f"  Per-request average:    {avg_per_request_tps:.2f} tok/s")
    print()
    print(f"  Cost (at ${HOURLY_COST}/hr): ${cost_for_run:.4f} for this run")
    print(f"  $/1M tokens:                ${cost_per_million:.2f}")
    print("=" * 60)

    if n_clients > 1:
        scaling = aggregate_tps / avg_per_request_tps if avg_per_request_tps > 0 else 0
        ideal = n_clients
        efficiency = (scaling / ideal * 100) if ideal > 0 else 0
        print(f"\n  Scaling efficiency: {efficiency:.0f}% of {n_clients}x ideal")
        if efficiency < 50:
            print(f"  → Single-thread bottleneck (engine is serial, lock contention)")
        elif efficiency > 90:
            print(f"  → Near-perfect concurrency (the lucky path)")
        else:
            print(f"  → Useful throughput scaling under contention")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8500",
                        help="mac-tensor UI server URL")
    parser.add_argument("--concurrent", "-c", type=int, default=4,
                        help="Number of concurrent clients")
    parser.add_argument("--requests", "-r", type=int, default=3,
                        help="Requests per client")
    parser.add_argument("--max-tokens", type=int, default=80,
                        help="Tokens per request")
    parser.add_argument("--hourly-cost", type=float, default=0.40,
                        help="Hourly cost of the cluster in USD (default 0.40 for 3x Scaleway M2)")
    args = parser.parse_args()
    main(args)
