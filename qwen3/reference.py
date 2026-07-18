#!/usr/bin/env python3
"""Transformers eager/BF16 reference used by checkpoint-dependent tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_reference(model_dir: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input-ids", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    args = parser.parse_args()
    import torch

    _, model = load_reference(args.model_dir)
    ids = [int(value) for value in args.input_ids.split()]
    with torch.no_grad():
        result = model.generate(
            torch.tensor([ids]),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )[0, len(ids):].tolist()
    print(json.dumps({"generated_ids": result}, separators=(",", ":")))


if __name__ == "__main__":
    main()
