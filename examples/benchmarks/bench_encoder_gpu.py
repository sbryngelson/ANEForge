"""Batched-GPU encoder baseline - the fair counterpart to the aneforge ANE batched
encoder (bench_encoder_batched.py: ~3978 embeds/sec @ B=16).

Runs the same MiniLM encoder on the Apple GPU (torch MPS), batched at the same B and
S, measuring embeds/sec + power, so encoder serving can be compared ANE-vs-GPU on
both throughput and efficiency (the LLM-decode question, asked for the workload the
ANE should actually win).

    python3 examples/benchmarks/bench_encoder_gpu.py
"""
import os, re, subprocess, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

NAME = "sentence-transformers/all-MiniLM-L6-v2"
S = 17  # match the ANE batched bench
_RAIL = {"ane": r"ANE Power:\s*([\d.]+)\s*mW", "cpu": r"CPU Power:\s*([\d.]+)\s*mW",
         "gpu": r"GPU Power:\s*([\d.]+)\s*mW"}


def main():
    have_sudo = subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    from transformers import AutoModel
    m = AutoModel.from_pretrained(NAME).eval().to(dev)
    ids = torch.randint(1, 30000, (1, S), dtype=torch.long)

    print(f"MiniLM encoder on {dev.upper()} (torch), S={S}\n")
    print(f"{'B':>3} | {'ms/batch':>9} | {'embeds/sec':>11} | {'per-stream ms':>13} | {'GPU mW':>7} | {'tok/s/W':>8}")
    base = None
    for B in [1, 2, 4, 8, 16]:
        x = ids.repeat(B, 1).to(dev)
        with torch.no_grad():
            for _ in range(5):
                _ = m(input_ids=x).last_hidden_state.mean(1); torch.mps.synchronize() if dev == "mps" else None
            pm = None
            if have_sudo:
                pm = subprocess.Popen(["sudo", "-n", "powermetrics", "--samplers", "ane_power,cpu_power,gpu_power",
                                       "--sample-rate", "100", "--sample-count", "60"],
                                      stdout=open(f"/tmp/pm_encgpu_{B}.log", "w"), stderr=subprocess.DEVNULL)
                time.sleep(0.3)
            t0 = time.perf_counter()
            for _ in range(50):
                _ = m(input_ids=x).last_hidden_state.mean(1)
                if dev == "mps":
                    torch.mps.synchronize()
            dt = (time.perf_counter() - t0) / 50
            if pm:
                pm.wait()
        eps_ = B / dt
        if base is None:
            base = eps_
        gpu_mw = active = float("nan")
        if pm:
            log = open(f"/tmp/pm_encgpu_{B}.log").read()
            rail = lambda r: (lambda v: sum(v) / len(v) if v else 0.0)([float(z) for z in re.findall(_RAIL[r], log)])
            gpu_mw = rail("gpu"); active = (rail("ane") + rail("cpu") + rail("gpu")) / 1000.0
        eff = eps_ / active if active and not np.isnan(active) and active > 0 else float("nan")
        print(f"{B:>3} | {dt*1e3:>9.2f} | {eps_:>11.0f} | {dt*1e3/B:>13.2f} | {gpu_mw:>7.0f} | {eff:>8.0f}")
    print("\nCompare to ANE batched encoder (bench_encoder_batched.py): ~2277 (B=1) -> 3978 (B=16) embeds/sec.")


if __name__ == "__main__":
    sys.exit(main())
