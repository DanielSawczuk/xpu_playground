#!/usr/bin/env python3
"""Exact-tokenization text frontend for the pure Level Zero Qwen3 engine."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy single-turn Qwen3-8B chat on BMG")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system")
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-seq-len", type=int, default=40960)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", action="store_true", dest="thinking")
    thinking.add_argument("--no-thinking", action="store_false", dest="thinking")
    parser.set_defaults(thinking=False)
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "qwen3_l0",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")
    if not 1 <= args.max_seq_len <= 40960:
        parser.error("--max-seq-len must be in [1, 40960]")
    return args


def main() -> int:
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "qwen3 chat: transformers==4.51.0 is required; "
            "install qwen3/requirements.txt",
            file=sys.stderr,
        )
        return 1

    # local_files_only is deliberate: model acquisition is always explicit.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    messages = []
    if args.system is not None:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=args.thinking,
    )
    if len(input_ids) + args.max_new_tokens > args.max_seq_len:
        print(
            f"qwen3 chat: tokenized request needs {len(input_ids) + args.max_new_tokens} "
            f"positions but --max-seq-len is {args.max_seq_len}",
            file=sys.stderr,
        )
        return 1
    if not args.engine.is_file():
        print(f"qwen3 chat: native engine not found: {args.engine}; run 'make qwen3'", file=sys.stderr)
        return 1
    command = [
        str(args.engine),
        "--model-dir", str(args.model_dir),
        "--input-ids", "-",
        "--max-new-tokens", str(args.max_new_tokens),
        "--max-seq-len", str(args.max_seq_len),
    ]
    try:
        completed = subprocess.run(
            command,
            input=" ".join(map(str, input_ids)) + "\n",
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        )
        payload = json.loads(completed.stdout)
        generated = payload["generated_ids"]
        if not isinstance(generated, list) or not all(isinstance(i, int) for i in generated):
            raise ValueError("generated_ids is not an integer list")
    except subprocess.CalledProcessError as error:
        return error.returncode or 1
    except (json.JSONDecodeError, KeyError, ValueError) as error:
        print(f"qwen3 chat: invalid native-engine output: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(tokenizer.decode(generated, skip_special_tokens=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
