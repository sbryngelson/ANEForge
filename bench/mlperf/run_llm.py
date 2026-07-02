#!/usr/bin/env python3
"""MLPerf-lite LLM-decode runner: greedy KV-cache decode on the ANE, reporting TTFT / TPOT / tokens-per-second
(the MLPerf LLM shape). Defaults to a cached Qwen3-0.6B; a synthetic prompt is used (per-token decode compute is
token-content-independent, so this measures the decode path without needing a tokenizer).

  PYTHONPATH=. python3 bench/mlperf/run_llm.py                      # cached Qwen3-0.6B, fp16
  PYTHONPATH=. python3 bench/mlperf/run_llm.py --int8 --gen 128
  PYTHONPATH=. python3 bench/mlperf/run_llm.py --llm ~/Models/Qwen3-0.6B --prompt 64 --gen 128

Writes a JSON to bench/mlperf/results/. NOT an official MLPerf submission (see README.md)."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # loadgen_lite, llm_decode

import loadgen_lite as lg      # noqa: E402
import llm_decode as ld        # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="MLPerf-lite LLM decode on the ANE")
    ap.add_argument("--llm", default=None, help="HF model name/path (default: cached Qwen3-0.6B)")
    ap.add_argument("--prompt", type=int, default=32, help="prompt length in tokens (synthetic ids)")
    ap.add_argument("--gen", type=int, default=64, help="tokens to generate")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--int8", action="store_true", help="int8 ANE weights")
    ap.add_argument("--out", default=None, help="results JSON path")
    args = ap.parse_args()

    compress = "int8" if args.int8 else None
    print(f"building LLM decode SUT ({compress or 'fp16'}, ANE) ...", flush=True)
    sut = ld.build_sut(model=args.llm, compress=compress, max_len=args.max_len)

    prompt = list(range(1, args.prompt + 1))
    r = lg.run_llm_decode(sut, prompt, gen_len=args.gen)
    print(r.summary(), flush=True)

    out = args.out or os.path.join(Path(__file__).resolve().parent, "results", f"llm_decode_{compress or 'fp16'}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model": sut.name, "compress": compress or "fp16", "result": r.to_dict()}, f, indent=2, allow_nan=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
