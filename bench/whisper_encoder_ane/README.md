# Whisper encoder on the Apple Neural Engine

The Whisper audio encoder, compiled with ANEForge into one program and run on the
Apple Neural Engine (no CoreML), benchmarked against the same encoder in PyTorch (CPU
and Metal/MPS) and against whisper.cpp's own Metal and CoreML encoders.

The encoder is the fixed-shape half of Whisper (an 80-channel log-mel over a fixed
1500-frame context), so it compiles once and dispatches many times. The
autoregressive decoder, whose KV-cache length changes every token, is not covered
here.

## Result (whisper-tiny encoder, trained weights, M-series)

| engine               | latency | energy / encode | fidelity      |
| -------------------- | ------: | --------------: | ------------- |
| ANE (ANEForge)       | 33.6 ms | ~107 mJ         | cosine 0.9998 |
| PyTorch-MPS (eager)  | 12.7 ms | ~173 mJ         | reference     |
| PyTorch CPU (fp32)   | 34.7 ms | --              | reference     |

The encoder runs on the ANE at correct fidelity (cosine 0.9998 on real speech, with an
identical transcript -- see Fidelity), but it is not fast: 33.6 ms is slower than the
GPU, and the energy is about 1.6x less than the PyTorch-MPS baseline (a modest win, and
measured against eager MPS, not an optimized Metal kernel, which would draw less). On
the same ANE, CoreML runs this encoder about 3x faster (see Compared to whisper.cpp);
ANEForge's generic attention lowering is the gap, and closing it is open work.

### Latency is weight-dependent

ANE execute time for this graph depends on the weight values, not only the shapes. The
trained checkpoint runs at ~33 ms; a randomly-initialised encoder of the same shape
runs at ~11 ms, because the trained weights peak about 5x higher in magnitude and push
the attention onto a slower path. Performance numbers here use the trained weights
(`bench_latency.py --real`, `bench_energy.py --real`); random init reports an optimistic
~3x and is not representative. Use `--real` for any latency or energy claim.

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
absorbs the gap on real speech. (Fidelity is robust to the weights; latency is not, as
noted above.)

## Compared to whisper.cpp

The PyTorch-MPS baseline above is an eager GPU path, not an optimized kernel.
whisper.cpp ships both an optimized Metal encoder and a CoreML encoder that reaches
the ANE, so it is the real point of comparison. Measured with its own `whisper-bench`
on the same tiny model, steady-state per encode:

| encoder            | latency  | hardware |
| ------------------ | -------: | -------- |
| whisper.cpp Metal  | ~7.3 ms  | GPU      |
| whisper.cpp CoreML | ~11.2 ms | ANE      |
| ANEForge           | ~33.6 ms | ANE      |

So for this encoder ANEForge is the slowest path: correct, but ~3x slower than CoreML
on the same ANE and ~5x slower than the Metal kernel. CoreML's converter lays attention
out in the ANE-native channels-first form (projections as 1x1 convolutions, heads split
along the channel axis, no transposes); ANEForge builds generic `[seq, d]` attention
with reshapes and transposes that map poorly to the ANE, and the trained weight
magnitudes expose the cost. The gap is in how the graph is lowered, not the hardware --
the same ANE does the encoder in 11 ms under CoreML. An ANE-native encoder layout in
ANEForge is open work.

Reproduce the whisper.cpp side: build it with and without `-DWHISPER_COREML=1` (the
CoreML encoder needs a converted `ggml-tiny-encoder.mlmodelc`), then
`whisper-bench -m models/ggml-tiny.bin`.

## Notes

- Energy is the ANE's only edge here, and a modest one: ~107 mJ vs PyTorch-MPS's
  ~173 mJ (1.6x), idle-subtracted whole-package. The ANE rail itself is low (~1.2 W),
  but the 33 ms runtime erases most of the per-encode advantage, and the measurement is
  against eager MPS, not a tuned Metal kernel.
- At seq 1500 `af.sdpa` decomposes to the same matmul/softmax as `af.mha`, because the
  native fused-attention layer is reliable only when the smaller attention axis is
  below 512. The two rows are identical by construction.
- The encoder has been wired into whisper.cpp end to end (a backend mirroring the
  CoreML seam, in a fork) and transcribes correctly; the in-engine encode runs at the
  same ~33 ms, confirming the latency is the encoder, not the integration.

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
