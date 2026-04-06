#!/usr/bin/env python3
"""
mac-tensor agent — Interactive agentic REPL backed by distributed expert nodes.

The model can call tools by emitting XML tags in its response. The agent
parses tool calls, executes them, and feeds the results back into the
conversation until the model produces a final answer.

Tools:
  <read>path</read>             — read a file
  <ls>path</ls>                 — list directory contents
  <shell>command</shell>        — run a shell command (read-only by default)
  <search>query</search>        — DuckDuckGo web search
  <write path="...">content</write>  — write a file (requires --write)
  <python>code</python>         — eval python expression

The model sees tool results in <result> tags and continues from there.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path


SYSTEM_PROMPT = """You are an autonomous coding agent running on a distributed Apple Silicon cluster. You can call tools by emitting XML tags in your response. The user will see your tool calls and their results, then you'll continue to the next step.

Available tools:

<read>path</read>
  Read a file. Path can be relative or absolute. Use ~ for home.
  Example: <read>~/projects/main.py</read>

<ls>path</ls>
  List directory contents. Use . for current directory.
  Example: <ls>.</ls>

<shell>command</shell>
  Run a read-only shell command. Avoid destructive operations.
  Example: <shell>git status</shell>

<search>query</search>
  Search the web via DuckDuckGo. Returns top results.
  Example: <search>MLX expert sniper benchmarks</search>

<python>code</python>
  Evaluate a Python expression (read-only, no imports beyond stdlib).
  Example: <python>len([1,2,3])</python>

How to use tools:
1. When you need information, emit ONE tool call and stop. Wait for the result.
2. After seeing the <result>, decide your next step.
3. When you have enough information, give the user a final answer in plain text (no tags).
4. Be concise. Don't narrate every step — just call tools and show results.

The user is asking:"""


# ============================================================
# TOOLS
# ============================================================


def tool_read(arg):
    """Read a file."""
    try:
        path = os.path.expanduser(arg.strip())
        with open(path) as f:
            content = f.read()
        # Cap at 8000 chars to avoid blowing the context
        if len(content) > 8000:
            content = content[:8000] + f"\n\n[... truncated, {len(content) - 8000} more chars ...]"
        return content
    except Exception as e:
        return f"Error reading {arg}: {e}"


def tool_ls(arg):
    """List directory."""
    try:
        path = os.path.expanduser(arg.strip())
        items = sorted(os.listdir(path))
        out = []
        for item in items[:100]:
            full = os.path.join(path, item)
            if os.path.isdir(full):
                out.append(f"  {item}/")
            else:
                size = os.path.getsize(full)
                out.append(f"  {item} ({size} bytes)")
        if len(items) > 100:
            out.append(f"  ... and {len(items) - 100} more")
        return "\n".join(out) if out else "(empty)"
    except Exception as e:
        return f"Error listing {arg}: {e}"


def tool_shell(arg, allow_write=False):
    """Run a shell command."""
    cmd = arg.strip()

    # Block obvious destructive commands unless --write
    if not allow_write:
        DANGEROUS = ["rm ", "rm\n", "rm\t", "mv ", "dd ", "mkfs", ">", ">>",
                     "chmod", "chown", "sudo", "kill", "shutdown", "reboot",
                     "format", "fdisk", ":(){"]
        for d in DANGEROUS:
            if d in cmd.lower():
                return f"Refused (potentially destructive: '{d}'). Pass --write to allow."

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=20
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if not out and not err:
            return f"(exit {result.returncode}, no output)"
        parts = []
        if out:
            if len(out) > 4000:
                out = out[:4000] + f"\n... [truncated]"
            parts.append(out)
        if err:
            if len(err) > 1000:
                err = err[:1000] + "..."
            parts.append(f"[stderr] {err}")
        if result.returncode != 0:
            parts.append(f"[exit {result.returncode}]")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return "Error: command timed out (20s limit)"
    except Exception as e:
        return f"Error: {e}"


def tool_search(arg):
    """DuckDuckGo HTML search."""
    query = arg.strip()
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Extract result links + snippets with regex
        results = []
        pattern = re.compile(
            r'class="result__a"[^>]*>([^<]+)</a>.*?'
            r'class="result__snippet"[^>]*>([^<]+)</a>',
            re.DOTALL,
        )
        for m in pattern.finditer(html)[:5]:
            title = re.sub(r"\s+", " ", m.group(1).strip())
            snippet = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            snippet = re.sub(r"\s+", " ", snippet)
            results.append(f"• {title}\n  {snippet[:200]}")

        if not results:
            return "No results"
        return "\n\n".join(results[:5])
    except Exception as e:
        return f"Search failed: {e}"


def tool_write(arg, content, allow_write=False):
    """Write a file."""
    if not allow_write:
        return "Refused. File writes require --write flag."
    try:
        path = os.path.expanduser(arg.strip())
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def tool_python(arg):
    """Eval a python expression."""
    try:
        # Restricted eval — no imports, no builtins beyond a safe set
        safe_builtins = {
            "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
            "chr": chr, "dict": dict, "divmod": divmod, "enumerate": enumerate,
            "filter": filter, "float": float, "hex": hex, "int": int,
            "len": len, "list": list, "map": map, "max": max, "min": min,
            "oct": oct, "ord": ord, "pow": pow, "range": range, "reversed": reversed,
            "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum,
            "tuple": tuple, "type": type, "zip": zip, "print": print,
        }
        result = eval(arg.strip(), {"__builtins__": safe_builtins}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ============================================================
# PARSER + AGENT LOOP
# ============================================================


TOOL_PATTERN = re.compile(
    r"<(read|ls|shell|search|python|write)(?:\s+([^>]*))?>(.+?)</\1>",
    re.DOTALL,
)


def parse_tool_call(text):
    """Find the FIRST tool call in the text. Returns (tool, args, content) or None."""
    m = TOOL_PATTERN.search(text)
    if not m:
        return None
    tool = m.group(1)
    attrs = m.group(2) or ""
    content = m.group(3)
    return (tool, attrs, content, m.start(), m.end())


def execute_tool(tool, attrs, content, allow_write=False):
    """Run a tool and return its result string."""
    if tool == "read":
        return tool_read(content)
    elif tool == "ls":
        return tool_ls(content)
    elif tool == "shell":
        return tool_shell(content, allow_write=allow_write)
    elif tool == "search":
        return tool_search(content)
    elif tool == "python":
        return tool_python(content)
    elif tool == "write":
        # Parse path attribute
        path_match = re.search(r'path="([^"]+)"', attrs)
        if not path_match:
            return "Error: <write> requires path attribute"
        return tool_write(path_match.group(1), content, allow_write=allow_write)
    else:
        return f"Unknown tool: {tool}"


# ============================================================
# DISTRIBUTED LLM BACKEND
# ============================================================


class DistributedBackend:
    """Wraps the distributed engine as a simple LLM call interface."""

    def __init__(self, model_key, node_urls):
        self.model_key = model_key
        self.node_urls = node_urls
        self.engine = None
        self.history = []  # list of (role, content) tuples

    def load(self):
        """Load the appropriate distributed engine."""
        # Add the parent dir to path so we can import the coordinator scripts
        script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        sys.path.insert(0, script_dir)

        if self.model_key == "gemma4":
            # Gemma 4 needs the custom model from cli-agent path
            cli_agent_src = os.path.expanduser("~/cli-agent/src")
            if os.path.exists(cli_agent_src):
                sys.path.insert(0, cli_agent_src)
            from gemma4_distributed import Gemma4DistributedEngine
            self.engine = Gemma4DistributedEngine(node_urls=self.node_urls)
        else:
            from distributed_interactive import InteractiveDistributedEngine
            self.engine = InteractiveDistributedEngine(node_urls=self.node_urls)

        self.engine.load()

    def chat(self, prompt, max_tokens=300, temperature=0.6):
        """Generate a response. The engine handles its own KV cache."""
        # We use the engine's generate() method which handles chat templates
        return self.engine.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def reset(self):
        self.engine.reset_cache()


# ============================================================
# MAIN AGENT LOOP
# ============================================================


def agent_loop(backend, max_iterations=8, max_tokens=400, allow_write=False):
    """Run the interactive agent REPL."""
    print()
    print("=" * 60)
    print("  mac-tensor agent")
    print(f"  Model: {backend.model_key} | Nodes: {len(backend.node_urls)}")
    print(f"  Tools: read, ls, shell, search, python" + (", write" if allow_write else ""))
    print("=" * 60)
    print()
    print("Type your question. Type 'reset' to clear context, 'quit' to exit.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            return
        if user_input.lower() == "reset":
            backend.reset()
            print("Context cleared.")
            continue

        # Build the agent prompt
        prompt = f"{SYSTEM_PROMPT} {user_input}"

        # Iterate: call model → parse tool → execute → feed back
        iteration = 0
        accumulated = ""

        while iteration < max_iterations:
            iteration += 1
            print(f"\n[Step {iteration}/{max_iterations}] Thinking...")

            try:
                response = backend.chat(prompt, max_tokens=max_tokens, temperature=0.5)
            except Exception as e:
                print(f"  ERROR: {e}")
                break

            # Look for a tool call in the response
            call = parse_tool_call(response)

            if call is None:
                # No tool call → final answer
                print("\nAgent:")
                print(response.strip())
                break

            tool, attrs, content, start, end = call

            # Show what the model decided to do
            preview = content[:80].replace("\n", " ")
            print(f"  → Calling <{tool}>: {preview}...")

            # Execute the tool
            result = execute_tool(tool, attrs, content, allow_write=allow_write)
            result_preview = result[:200].replace("\n", " | ")
            print(f"  ← Result: {result_preview}{'...' if len(result) > 200 else ''}")

            # Append to the conversation: model's request + tool result
            tool_call_text = response[start:end]
            prompt = (
                f"{prompt}\n{response[:end]}\n"
                f"<result>\n{result}\n</result>\n"
                f"Continue: either call another tool or give the final answer."
            )

            # Reset the engine's cache between tool calls so it processes the
            # full updated prompt fresh (otherwise the KV cache is stale)
            backend.reset()
        else:
            print(f"\n[Hit {max_iterations} iteration limit]")


def main(args):
    """Entry point called from cli.py."""
    if not args.nodes:
        print("Error: --nodes is required")
        print("Example: mac-tensor agent --model gemma4 --nodes http://mac2:8401,http://mac3:8401")
        sys.exit(1)

    node_urls = [u.strip() for u in args.nodes.split(",")]
    model_key = args.model or "gemma4"

    print(f"Loading {model_key} distributed engine...")
    backend = DistributedBackend(model_key=model_key, node_urls=node_urls)
    backend.load()

    agent_loop(
        backend,
        max_iterations=args.max_iterations or 8,
        max_tokens=args.max_tokens or 400,
        allow_write=args.write,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen35", "gemma4"], default="gemma4")
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--write", action="store_true",
                        help="Allow file writes and destructive shell commands")
    args = parser.parse_args()
    main(args)
