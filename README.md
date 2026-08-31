# ClinConsensus

Portable evaluation, validation, training, and local-judge utilities for the
ClinConsensus benchmark and ClinConsensus-Judge-8B.

## Repositories

- Dataset: `https://huggingface.co/datasets/skylenage/ClinConsensus`
- Model: `https://huggingface.co/skylenage/ClinConsensus-Judge-8B`
- Code: `https://github.com/crzzx1/ClinConsensus`

The public dataset contains only the 900-case low-difficulty tier and excludes
reference answers. No API keys, provider transcripts, physician identifiers,
or private training examples are distributed here.

## Install

Metric and validation scripts use only the Python standard library. Local model
inference and training require packages in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

## Validate the public dataset

```bash
python scripts/validate_dataset.py \
  --data /path/to/clinconsensus_low.jsonl \
  --strict-contact
```

## Score rubric judgments

```bash
python scripts/eval_cacs.py \
  --data /path/to/clinconsensus_low.jsonl \
  --judgements-dir /path/to/judge_results \
  --output-dir results/run \
  --threshold 10
```

Each judgment file is a JSON list aligned to the public case order (or carrying
resolvable `case_id` values), with 30 Boolean `criteria_met` decisions per
case. The evaluator writes case-level and aggregate CSV files plus an input
hash manifest.

## Run the released judge locally

Prepare JSONL rows with `user_message`, `response`, and `rubric_item`, then run:

```bash
python scripts/run_local_judge.py \
  --model skylenage/ClinConsensus-Judge-8B \
  --input items.jsonl \
  --output predictions.jsonl
```

The model is an evaluation grader, not a medical assistant. Its outputs must
not be interpreted as medical advice or clinical deployment certification.

## Training reproduction

`scripts/train_sft_judge.py` exposes the reported Qwen3-8B full-parameter SFT
configuration. The physician-labeled training records are not included in the
public repository; access is governed separately.

## Licenses

Code is MIT licensed (`LICENSE`). Dataset terms are documented separately in
`LICENSE-DATA`. Model terms are provided with the model repository.
