# Whisper encoder on the Apple Neural Engine

The Whisper audio encoder, compiled with ANEForge into one program and run on the
Apple Neural Engine (no CoreML), benchmarked against the same encoder in PyTorch on
the CPU and the Metal GPU (MPS).

The encoder is the fixed-shape half of Whisper (an 80-channel log-mel over a fixed
1500-frame context), so it compiles once and dispatches many times. The
autoregressive decoder, whose KV-cache length changes every token, is not covered
here.

## Result (whisper-tiny encoder, M-series)

| engine          | latency | energy / encode | fidelity      |
| --------------- | ------: | --------------: | ------------- |
| ANE             | 11.4 ms | ~32 mJ          | cosine 0.9998 |
| Metal GPU (MPS) | 13.0 ms | ~126 mJ         | reference     |
| CPU (fp32)      | 36.1 ms | --              | reference     |

The ANE matches the GPU on latency and uses about 3.8x less energy per encode, at
cosine 0.9998 on real speech against the PyTorch reference, with an identical
transcript (see Fidelity below). For reference, whisper.cpp's own Metal encoder runs
the same tiny model at about 14.3 ms on this machine, so the ANE encoder is faster
than the engine it would replace, not only the PyTorch baseline. The latency and the
energy ratio are stable across runs; absolute milliJoules move with background system
load.

## Fidelity

A randomly-initialised encoder reads cosine 0.999987. The trained checkpoint reads
cosine 0.9998 on real speech (jfk.wav) and 0.9977 on the harder synthetic signal in
`validate_real.py` (6.8% relative error). The gap is the ANE's gelu LUT and fp16
losing more on the sharp, high-dynamic-range activations a trained encoder produces;
it needs both real weights and real input to appear, so a synthetic check alone reads
the optimistic number. `fidelity.py` measures the random-init case; `validate_real.py`
the trained checkpoint.

The feature error does not change the transcript. Decoding the ANE encoder output with
the HF Whisper decoder transcribes jfk.wav identically to the reference encoder
(`wer_proxy.py`). That is one clip, not a dataset-wide WER, but it shows the decoder
absorbs the gap on real speech. Latency and energy are weight-independent and unchanged
by which weights run.

## Notes

- The advantage is energy, not latency. Latency is close; the ANE draws about 1.9 W
  on its own rail against the GPU's 8 W (idle-subtracted, whole package).
- At seq 1500 `af.sdpa` decomposes to the same matmul/softmax as `af.mha`, because the
  native fused-attention layer is reliable only when the smaller attention axis is
  below 512. The two rows are identical by construction.
- The main table's GPU baseline is PyTorch-eager MPS, the same baseline the top-level
  benchmarks use. whisper.cpp's own Metal encoder, measured separately on the same
  tiny model and clip, runs at about 14.3 ms, also slower than the ANE here. The
  comparison still missing is whisper.cpp's CoreML path, which can already reach the
  ANE for the encoder; measuring against it, and a real in-engine integration, need a
  whisper.cpp fork.

## C++ runner

`whisper_ane_run.cpp` loads the compiled encoder and dispatches it on the ANE from
C++ with no Python, as a whisper.cpp backend would at model load.

The compiled on-device program is keyed to the OS build and is not a shippable binary,
and the cached bundle is not cold-loadable by a fresh process:
`program_library_create` succeeds, but creating the precompiled operation fails with
"Must re-compile the E5 bundle". The reuse path is therefore compile-with-cache: ship
the portable `model.mil` and `weights.bin`, and compile once per process at model load
(about 0.6 s, cached after). The runner uses the existing `ane_e5rt_program_compile`
entry point, so ANEForge is unchanged.

## Files

| file                  | what it does                                                        |
| --------------------- | ------------------------------------------------------------------ |
| `encoder.py`          | the whisper-tiny encoder built two ways (HF reference + ANEForge)  |
| `fidelity.py`         | cosine and relative error vs the reference (random-init encoder)    |
| `validate_real.py`    | the same, on the trained whisper-tiny checkpoint and a real log-mel |
| `wer_proxy.py`        | ANE-encoder transcript vs reference transcript on real speech       |
| `bench_latency.py`    | ANE (mha, sdpa) vs CPU vs MPS latency                              |
| `bench_energy.py`     | ANE vs MPS energy via powermetrics (idle-subtracted, whole-package)|
| `export_bundle.py`    | compile and persist the program, dump fp16 vectors and a manifest  |
| `whisper_ane_run.cpp` | standalone C++ ANE runner (links the dispatch dylib)               |
| `run_cpp.sh`          | export, build, and run the C++ runner end to end                   |

## Run

From the repo root:

```sh
PYTHONPATH=. python3 bench/whisper_encoder_ane/fidelity.py
PYTHONPATH=. python3 bench/whisper_encoder_ane/validate_real.py   # downloads whisper-tiny
PYTHONPATH=. python3 bench/whisper_encoder_ane/wer_proxy.py       # downloads whisper-tiny + jfk.wav
PYTHONPATH=. python3 bench/whisper_encoder_ane/bench_latency.py
PYTHONPATH=. python3 bench/whisper_encoder_ane/bench_energy.py    # needs passwordless sudo
sh bench/whisper_encoder_ane/run_cpp.sh
```

The encoder is randomly initialised (fixed seed), so nothing is downloaded. To run the
real checkpoint, load whisper-tiny's encoder weights into the state dict in
`encoder.py`; the key names already match HF's `WhisperEncoder`.
