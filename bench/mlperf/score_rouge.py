#!/usr/bin/env python3
"""Canonical ROUGE for a llm_summarize result JSON, scored OUT OF PROCESS from the ANE.

`llm_summarize.py` writes {summary, reference} per row using a self-contained inline ROUGE. The canonical
`rouge_score` (Porter stemmer, as MLPerf uses) cannot run in the same process as an ANE dispatch -- it segfaults
-- so re-score here; this script does NOT import aneforge.

  pip install rouge-score
  python3 bench/mlperf/score_rouge.py bench/mlperf/results/llm_summarize_fp16_n25.json
"""
import json
import sys

from rouge_score import rouge_scorer

_KEYS = ("rouge1", "rouge2", "rougeL", "rougeLsum")


def main(path):
    rows = [r for r in json.load(open(path))["rows"] if "reference" in r]
    if not rows:
        print(f"{path}: no rows carry a reference (re-run llm_summarize.py to store them)"); return 1
    scorer = rouge_scorer.RougeScorer(list(_KEYS), use_stemmer=True)
    agg = dict.fromkeys(_KEYS, 0.0)
    for r in rows:
        s = scorer.score(r["reference"], r["summary"])
        for k in _KEYS:
            agg[k] += s[k].fmeasure
    n = len(rows)
    print(f"canonical rouge_score (Porter stemmer), {n} summaries from {path}:")
    for k in _KEYS:
        print(f"  {k:10s} {agg[k] / n:.4f}")
    print("(MLPerf small-LLM edge gate is ~99% of the reference model's ROUGE; the full edge set is 5000 articles)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "bench/mlperf/results/llm_summarize_fp16_n25.json"))
