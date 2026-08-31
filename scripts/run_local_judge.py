#!/usr/bin/env python3
"""Run ClinConsensus-Judge-8B on rubric-level JSONL inputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT = """Your job is to evaluate whether the assistant's response meets the criteria of a specific rubric item.

# User Query (the input the model received)
{user_message}

# Assistant Response (the model's output to evaluate)
{response}

# Rubric Item to Evaluate
{rubric_item}

# Instructions
Evaluate whether the assistant's response meets the criteria described in the rubric item above.

Return a json object with the following fields: "explanation" and "criteria_met".
- "explanation": A concise explanation (in Chinese) of why the response does or does not meet the criteria.
- "criteria_met": A boolean (true/false) indicating whether the response meets the criteria.

Important rules:
1. If a rubric item has multiple criteria or sentences, ALL must be met for criteria_met to be true. If any single criterion is not met, return false.
2. If a criterion uses cue words such as "如", "例如", "比如", "包括", "such as", "for example", or "including", the response does not need to include every listed example; it only needs to address the general requirement adequately.
3. Judge factual medical accuracy. If the rubric specifies a drug, dosage, guideline, or medical fact, the response must be consistent with current medical knowledge.
4. If the rubric asks "是否" (whether), evaluate strictly: the response must clearly demonstrate the required behavior.

# Output Format
Return ONLY a JSON object in a markdown code block:
```json
{{"explanation": "...", "criteria_met": true/false}}
```"""


def parse_output(text: str) -> tuple[bool | None, bool]:
    match = re.search(r'"criteria_met"\s*:\s*(true|false)', text, re.IGNORECASE)
    return ((match.group(1).lower() == "true"), True) if match else (None, False)


def batches(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=5120)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    required = {"user_message", "response", "rubric_item"}
    for index, record in enumerate(records, 1):
        missing = required - record.keys()
        if missing:
            raise SystemExit(f"input row {index} is missing: {sorted(missing)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for record_batch in batches(records, args.batch_size):
            prompts = []
            for record in record_batch:
                content = PROMPT.format(**{key: record[key] for key in required})
                prompts.append(tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ))
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_length,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            prompt_width = inputs["input_ids"].shape[1]
            texts = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
            for record, raw in zip(record_batch, texts):
                label, parse_ok = parse_output(raw)
                result = dict(record)
                result.update({"criteria_met": label, "parse_ok": parse_ok, "judge_output": raw})
                output.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
