#!/usr/bin/env python3
"""
MLX server — drop-in replacement for llama-server.
Runs Qwen3.5 on Apple MLX framework for ~2x speed vs llama.cpp.

Usage:
    pip3 install mlx-lm
    python3 mlx_server.py              # 9B model
    python3 mlx_server.py --model 35b  # 35B MoE model

This starts an OpenAI-compatible API at localhost:8000.
The agent.py works with either llama-server or mlx_server.py.
"""

import argparse
import sys
import os
import subprocess

MODELS = {
    "9b": "mlx-community/Qwen3.5-9B-4bit",
    "35b": "mlx-community/Qwen3.5-35B-A3B-4bit",  # MoE model
}

def main():
    parser = argparse.ArgumentParser(description="MLX server for mac code")
    parser.add_argument("--model", default="9b", choices=["9b", "35b"],
                       help="Which model to run (default: 9b)")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    model_id = MODELS[args.model]

    # Check if mlx-lm is installed
    try:
        import mlx_lm
    except ImportError:
        print("MLX not installed. Run:")
        print("  pip3 install mlx-lm")
        sys.exit(1)

    print(f"\n  🍎 mac code MLX server")
    print(f"  Model: {model_id}")
    print(f"  Port:  {args.port}")
    print(f"  MLX gives ~2x speed vs llama.cpp on Apple Silicon")
    print()

    # Start mlx_lm.server which provides OpenAI-compatible API
    cmd = [
        sys.executable, "-m", "mlx_lm.server",
        "--model", model_id,
        "--port", str(args.port),
    ]

    print(f"  Starting: {' '.join(cmd)}")
    print()

    os.execvp(sys.executable, cmd)

if __name__ == "__main__":
    main()
