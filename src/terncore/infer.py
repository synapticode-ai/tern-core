"""
infer.py — CLI entry point for ternary inference.

Usage:
    python -m terncore.infer

Wraps the high-level inference_api.generate() function for command-line use.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="terncore.infer",
        description="Run ternary inference with TinyLlama",
    )
    parser.add_argument("prompt", nargs="?", default="Hello, world!")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    from terncore.inference_api import generate

    result = generate(args.prompt, max_tokens=args.max_tokens)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(result.text)


if __name__ == "__main__":
    main()
