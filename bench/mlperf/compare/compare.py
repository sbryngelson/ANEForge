#!/usr/bin/env python3
"""Place the ANE's MLPerf ResNet-50 numbers next to published competitors.

Reads `competitors.csv` (each row carries its MLPerf round + source path) and renders a markdown table plus
a data-derived summary. The ANE row is self-measured and UNOFFICIAL; competitor rows are official MLPerf
Inference results from the public `mlcommons/inference_results_*` repos.

  python3 bench/mlperf/compare/compare.py [--out FILE.md]"""
from __future__ import annotations
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _fmt_ms(v):
    return f"{float(v):.3f}" if v else "--"


def _fmt_qps(v):
    return f"{float(v):,.0f}" if v else "--"


def _table(rows):
    head = ("System", "Category", "MLPerf round", "SingleStream p90 (ms)", "Offline (samples/s)", "Power class")
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for r in rows:
        official = r["official"].strip().lower() in ("yes", "y", "true", "1")
        name = r["system"] + ("" if official else " *(unofficial -- self-measured)*")
        out.append("| " + " | ".join((
            name, r["category"], r["mlperf_round"],
            _fmt_ms(r["singlestream_ms"]), _fmt_qps(r["offline_qps"]), r["power_class"])) + " |")
    return "\n".join(out)


def _summary(rows):
    """Comparison lines derived from the numbers."""
    ane = next((r for r in rows if r["official"].strip().lower() in ("no", "n", "false", "0")), None)
    if not ane:
        return ""
    lines = []
    a_ss, a_off = float(ane["singlestream_ms"]), float(ane["offline_qps"])
    peers = [r for r in rows if r is not ane and r["category"] == "Edge" and r["singlestream_ms"]]
    for p in peers:
        p_ss, p_off = float(p["singlestream_ms"]), float(p["offline_qps"])
        lines.append(
            f"- vs {p['system']} ({p['mlperf_round']}): SingleStream {a_ss:.3f} ms vs {p_ss:.3f} ms "
            f"({a_ss / p_ss:.2f}x its latency); Offline {a_off:,.0f} vs {p_off:,.0f} samples/s "
            f"({a_off / p_off:.2f}x its throughput).")
    dc = [r for r in rows if r["category"] == "Datacenter" and r["offline_qps"]]
    for d in dc:
        d_off = float(d["offline_qps"])
        lines.append(
            f"- vs {d['system']} ({d['mlperf_round']}, datacenter): Offline {a_off:,.0f} vs {d_off:,.0f} "
            f"samples/s ({d_off / a_off:.0f}x the ANE) -- a different power and cost class.")
    return "\n".join(lines)


def render(rows):
    parts = [
        "### MLPerf ResNet-50: the ANE next to published competitors",
        "",
        _table(rows),
        "",
        "**Summary (from the numbers):**",
        _summary(rows),
        "",
        "**Reading this table**",
        "- The ANE row is **unofficial**: self-measured with `bench/mlperf` under the real MLCommons LoadGen "
        "(Result is: VALID), with no MLCommons audit trail. Competitor rows are **official** MLPerf Inference "
        "results from the public `mlcommons/inference_results_*` repos (source path per row in "
        "`competitors.csv`).",
        "- **SingleStream** is latency-bound at batch 1; **Offline** rewards batching, which the batch-1 ANE "
        "program does not do, so it trails there.",
        "- **Power class** is each part's device power envelope (SoC block / module TDP / board TDP), not a "
        "MLPerf Power measurement.",
        "- **Rounds differ**: ResNet-50 edge submissions from the big vendors thinned out after ~v3.1 (NVIDIA "
        "moved Jetson to LLM / Stable Diffusion), so the most recent official ResNet-50 Orin result is v3.1. "
        "ResNet-50 itself is unchanged across rounds.",
        "",
        "Sources: " + "; ".join(sorted({r["source"] for r in rows})),
    ]
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Compare the ANE's MLPerf ResNet-50 to published competitors.")
    ap.add_argument("--csv", default=os.path.join(HERE, "competitors.csv"))
    ap.add_argument("--out", default=None, help="also write the rendered markdown here")
    args = ap.parse_args()

    rows = _load(args.csv)
    md = render(rows)
    print(md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
