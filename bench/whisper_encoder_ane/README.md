# Whisper encoder on the Apple Neural Engine

Whisper's audio encoder, compiled with ANEForge into one program and run **directly
on the Apple Neural Engine** (no CoreML), then compared against the same encoder in
PyTorch on the CPU and the Metal GPU (MPS).

The encoder is the heavy, fixed-shape half of Whisper (an 80-channel log-mel spans a
fixed 1500-frame context), which makes it a clean ahead-of-time target: compile once,
dispatch many times. The autoregressive decoder is a separate problem (its KV-cache
length changes every token) and is not addressed here.

## Result (whisper-tiny encoder, M-series)

| engine            | latency   | energy / encode | fidelity     |
| ----------------- | --------: | --------------: | ------------ |
| **ANE**           | 11.4 ms   | ~32 mJ          | cosine 1.0000 |
| Metal GPU (MPS)   | 13.0 ms   | ~126 mJ         | reference    |
| CPU (fp32)        | 36.1 ms   | --              | reference    |

The ANE is slightly faster than the GPU and uses about **3.8x less energy per
encode**, at cosine 0.999987 against the PyTorch reference. Latency and the energy
ratio are stable across runs; the absolute milliJoule figures move with background
system load (the ratio does not).

## What this shows, and what it does not

- **Fidelity is exact for practical purposes.** cosine 0.999987 over the full stack,
  including the gelu LUT the ANE uses, on a randomly-initialised encoder. Latency and
  energy are weight-independent, so the numbers carry to the published checkpoint.
- **The win is energy, not raw speed.** Latency is close (the GPU is fast on this
  workload); the ANE's advantage is drawing ~1.9 W on its own rail versus the GPU's
  ~8 W (idle-subtracted, whole package).
- **The native fused-attention layer does not apply here.** At seq 1500, `af.sdpa`
  decomposes to the same matmul/softmax as `af.mha` (the native layer is reliable only
  when the smaller attention axis is < 512). The `mha` and `sdpa` rows are identical
  by construction.
- **The baseline is PyTorch-eager MPS, not whisper.cpp's hand-tuned Metal kernel**, and
  not whisper.cpp's existing CoreML path (which can already reach the ANE for the
  encoder). A true head-to-head needs a whisper.cpp fork; this measures the ANEForge
  path against a standard PyTorch GPU baseline, the same baseline the project's
  top-level benchmarks use.

## Integration shape (the C++ runner)

`whisper_ane_run.cpp` loads the compiled encoder and dispatches it on the ANE from
plain C++ with no Python, which is what a whisper.cpp backend would do at model load.

The compiled on-device program is keyed to the OS build and is not a shippable binary
artifact, and the cached bundle is not cold-loadable by a fresh process
(`program_library_create` succeeds, but creating the precompiled operation fails with
"Must re-compile the E5 bundle"). So the supported reuse path is **compile-with-cache**:
ship the portable `model.mil` + `weights.bin`, and compile once per process at model
load (about 0.6 s, cached thereafter). The runner uses the existing
`ane_e5rt_program_compile` entry point; no changes to ANEForge are needed.

## Files

| file                 | what it does                                                        |
| -------------------- | ------------------------------------------------------------------- |
| `encoder.py`         | the whisper-tiny encoder built two ways (HF reference + ANEForge)   |
| `fidelity.py`        | cosine / relative error vs the PyTorch reference                    |
| `bench_latency.py`   | ANE (mha, sdpa) vs CPU vs MPS latency                               |
| `bench_energy.py`    | ANE vs MPS energy via powermetrics (idle-subtracted, whole-package) |
| `export_bundle.py`   | compile + persist the program, dump fp16 vectors + a port manifest  |
| `whisper_ane_run.cpp`| standalone C++ ANE runner (links the dispatch dylib)                |
| `run_cpp.sh`         | export, build, and run the C++ runner end to end                    |

## Run

From the repo root:

```sh
PYTHONPATH=. python3 bench/whisper_encoder_ane/fidelity.py        # cosine 0.999987
PYTHONPATH=. python3 bench/whisper_encoder_ane/bench_latency.py   # ANE vs CPU vs MPS
PYTHONPATH=. python3 bench/whisper_encoder_ane/bench_energy.py    # needs passwordless sudo
sh bench/whisper_encoder_ane/run_cpp.sh                           # C++, on the ANE, no Python
```

The encoder is randomly initialised (fixed seed) so nothing is downloaded. To
reproduce against the real checkpoint, load whisper-tiny's encoder weights into the
state dict in `encoder.py` (the key names already match HF's `WhisperEncoder`).
