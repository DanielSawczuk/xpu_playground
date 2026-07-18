#!/usr/bin/env python3
"""Checkpoint-dependent tokenizer, cache, layer, and greedy-token checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import unittest


MODEL_DIR = os.environ.get("MODEL_DIR")
ENGINE = Path(os.environ.get("QWEN3_ENGINE", "./qwen3_l0")).resolve()


@unittest.skipUnless(MODEL_DIR, "set MODEL_DIR to the pinned Qwen3-8B checkpoint")
class CheckpointConformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cls.torch = torch
        cls.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, local_files_only=True, trust_remote_code=False
        )
        cls.model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).eval()

    def chat_ids(self, text: str, thinking: bool = False):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )

    def test_tokenizer_unicode_specials_thinking_and_round_trip(self):
        for text in ["plain ASCII", "雪と🌨️", "مرحبا", "e\u0301 vs é"]:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            self.assertEqual(self.tokenizer.encode(
                self.tokenizer.decode(ids), add_special_tokens=False), ids)
        self.assertEqual(self.tokenizer.eos_token_id, 151645)
        no_thinking = self.chat_ids("hello", False)
        thinking = self.chat_ids("hello", True)
        self.assertNotEqual(no_thinking, thinking)
        self.assertIn(151644, no_thinking)  # <|im_start|>
        self.assertIn(151645, [self.tokenizer.eos_token_id])

    def test_all_36_blocks_and_updated_cache_are_finite(self):
        torch = self.torch
        ids = torch.tensor([[151644, 872, 198]])
        with torch.no_grad():
            first = self.model(
                input_ids=ids, use_cache=True, output_hidden_states=True,
                return_dict=True,
            )
        self.assertEqual(len(first.hidden_states), 37)
        self.assertEqual(len(first.past_key_values), 36)
        for layer, state in enumerate(first.hidden_states[1:]):
            self.assertTrue(torch.isfinite(state.float()).all(), f"layer {layer}")
        with torch.no_grad():
            second = self.model(
                input_ids=torch.tensor([[42]]),
                past_key_values=first.past_key_values,
                use_cache=True,
                return_dict=True,
            )
        self.assertEqual(len(second.past_key_values), 36)
        for layer in range(36):
            key, value = second.past_key_values[layer]
            self.assertEqual(key.shape[-2], ids.shape[-1] + 1)
            self.assertEqual(value.shape[-2], ids.shape[-1] + 1)
            self.assertTrue(torch.isfinite(key.float()).all(), f"K cache layer {layer}")
            self.assertTrue(torch.isfinite(value.float()).all(), f"V cache layer {layer}")

    @unittest.skipUnless(ENGINE.is_file(), "build qwen3_l0 or set QWEN3_ENGINE")
    def test_prompt_and_cached_greedy_ids_match_transformers(self):
        torch = self.torch
        for prompt in ["Reply with one word: blue", "Unicode: 雪"]:
            ids = self.chat_ids(prompt, False)
            with torch.no_grad():
                expected = self.model.generate(
                    torch.tensor([ids]),
                    max_new_tokens=4,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=[151645, 151643],
                    pad_token_id=151645,
                )[0, len(ids):].tolist()
            completed = subprocess.run(
                [str(ENGINE), "--model-dir", str(MODEL_DIR),
                 "--input-ids", "-", "--max-new-tokens", "4"],
                input=" ".join(map(str, ids)) + "\n",
                text=True,
                stdout=subprocess.PIPE,
                check=True,
            )
            self.assertEqual(json.loads(completed.stdout)["generated_ids"], expected)


if __name__ == "__main__":
    unittest.main()
