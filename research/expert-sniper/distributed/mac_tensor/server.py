#!/usr/bin/env python3
"""
mac-tensor ui — Web chat UI for the distributed agent.

Serves a single-page HTML chat interface and a Server-Sent Events
endpoint that streams agent events (steps, tool calls, results, final answer).

Usage:
    mac-tensor ui --model gemma4 --nodes http://mac2:8401,http://mac3:8401
    # Then open http://localhost:8500 in your browser
"""

import json
import os
import sys
import time
import threading
from queue import Queue, Empty


def run_server(model_key, node_urls=None, host="0.0.0.0", port=8500, allow_write=False,
               vision=False, stream_dir=None, source_dir=None):
    """Start the FastAPI server with the agent backend pre-loaded.

    Two modes:
      - Distributed text-only: pass node_urls
      - Single-machine vision: pass vision=True + stream_dir + source_dir
    """
    from fastapi import FastAPI, Request, UploadFile, File, Form
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
    from .agent import AgentBackend, run_agent_turn_stream

    vision_engine = None

    if vision:
        print(f"Loading vision Gemma 4 sniper (single-machine)...")
        from .vision_engine import VisionGemma4Sniper
        vision_engine = VisionGemma4Sniper(
            stream_dir=stream_dir or "~/models/gemma4-stream",
            source_dir=source_dir or "~/models/gemma4-26b-4bit",
        )
        vision_engine.load()
        print("Vision engine ready.")
        backend = None  # Not used in vision mode
    else:
        print(f"Loading {model_key} distributed engine...")
        backend = AgentBackend(model_key=model_key, node_urls=node_urls)
        backend.load()
        print(f"Backend ready. Connected to {len(node_urls)} expert nodes.")

    app = FastAPI(title="mac-tensor agent UI")

    # Read the static HTML file shipped alongside this server
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    html_path = os.path.join(static_dir, "chat.html")
    with open(html_path) as f:
        chat_html = f.read()

    # Inject backend info into the HTML so the UI can show it
    if vision:
        model_label = "Gemma 4-26B-A4B (Vision)"
        node_count_label = "single Mac · vision enabled"
    else:
        model_label = {"gemma4": "Gemma 4-26B-A4B",
                       "qwen35": "Qwen 3.5-35B-A3B"}.get(model_key, model_key)
        node_count_label = f"{len(node_urls)} expert nodes"

    chat_html = chat_html.replace("{{MODEL_NAME}}", model_label) \
                          .replace("{{NODE_COUNT}}", node_count_label) \
                          .replace("{{VISION_ENABLED}}", "true" if vision else "false")

    # Lock so only one chat request runs at a time (single MoE engine)
    lock = threading.Lock()

    @app.get("/")
    async def index():
        return HTMLResponse(chat_html)

    @app.get("/api/info")
    async def info():
        return {
            "model": model_key,
            "nodes": node_urls,
            "allow_write": allow_write,
            "vision": vision,
        }

    @app.post("/api/reset")
    async def reset():
        with lock:
            if vision_engine:
                vision_engine.sniper.reset_cache()
            elif backend:
                backend.reset()
        return {"ok": True}

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        message = body.get("message", "").strip()
        if not message:
            return JSONResponse({"error": "empty message"}, status_code=400)

        max_iterations = int(body.get("max_iterations", 5))
        max_tokens = int(body.get("max_tokens", 300))

        def event_stream():
            with lock:
                try:
                    for event in run_agent_turn_stream(
                        backend, message,
                        max_iterations=max_iterations,
                        max_tokens=max_tokens,
                        allow_write=allow_write,
                    ):
                        yield f"data: {json.dumps(event)}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"},
        )

    @app.post("/api/chat_vision")
    async def chat_vision(
        message: str = Form(...),
        max_tokens: int = Form(200),
        image: UploadFile = File(None),
    ):
        """Vision chat endpoint — accepts an optional image upload."""
        if vision_engine is None:
            return JSONResponse({"error": "vision mode not enabled"}, status_code=400)

        # Save uploaded image to a temp file
        image_path = None
        if image is not None and image.filename:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix="_" + image.filename, delete=False)
            tmp.write(await image.read())
            tmp.close()
            image_path = tmp.name

        def event_stream():
            with lock:
                try:
                    yield f"data: {json.dumps({'type': 'step_start', 'step': 1, 'max': 1})}\n\n"

                    chunks = []
                    def on_chunk(text):
                        chunks.append(text)

                    output = vision_engine.generate(
                        message,
                        image_path=image_path,
                        max_tokens=max_tokens,
                        temperature=0.6,
                        on_chunk=on_chunk,
                    )

                    for chunk in chunks:
                        yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"

                    yield f"data: {json.dumps({'type': 'final', 'text': output.strip()})}\n\n"
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                finally:
                    # Clean up temp image
                    if image_path and os.path.exists(image_path):
                        try:
                            os.unlink(image_path)
                        except Exception:
                            pass

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"},
        )

    print()
    print("=" * 60)
    print(f"  mac-tensor UI ready")
    print(f"  Open: http://localhost:{port}")
    print(f"        http://{_local_ip()}:{port}  (LAN access)")
    print("=" * 60)
    print()

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


def _local_ip():
    """Best-effort detection of the LAN IP."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def main(args):
    vision = getattr(args, "vision", False)

    if vision:
        # Vision mode: single-machine, no distributed nodes needed
        run_server(
            model_key="gemma4",
            node_urls=None,
            host=args.host or "0.0.0.0",
            port=args.port or 8500,
            allow_write=getattr(args, "write", False),
            vision=True,
            stream_dir=getattr(args, "stream_dir", None),
            source_dir=getattr(args, "source_dir", None),
        )
    else:
        if not args.nodes:
            print("Error: --nodes is required (or pass --vision for single-machine mode)")
            sys.exit(1)
        node_urls = [u.strip() for u in args.nodes.split(",")]
        run_server(
            model_key=args.model or "gemma4",
            node_urls=node_urls,
            host=args.host or "0.0.0.0",
            port=args.port or 8500,
            allow_write=args.write,
        )
