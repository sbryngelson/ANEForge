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
| ANE (ANEForge)       | 9.5 ms  | ~30 mJ          | cosine 0.9998 |
| PyTorch-MPS (eager)  | 12.7 ms | ~160 mJ         | reference     |
| PyTorch CPU (fp32)   | 34.7 ms | --              | reference     |

The encoder runs on the ANE at 9.5 ms per encode and about 5x less energy than the
PyTorch-MPS baseline, at cosine 0.9998 on real speech with an identical transcript (see
Fidelity). On the same ANE, this is faster than CoreML's own encoder (~11 ms) and the
energy edge holds even against an optimized Metal kernel (see Compared to whisper.cpp).

This needed the ANE-native layout. A direct `[seq, d]` translation of the encoder runs
at ~33 ms (3x slower, and slower than the GPU); the speed comes from keeping the whole
stack channels-first.

### The ANE-native layout

The fast encoder (`build_cf`) keeps every tensor in `[1, d_model, 1, S]`, the layout
the ANE is built for: projections are 1x1 convolutions, attention is `einsum` over the
channel axis, and the norm is `channel_layer_norm` (LayerNorm over channels, added to
ANEForge for this). There are no `[seq, d]` transposes anywhere. The generic version
(`build`) instead reshapes to `[seq, d]` per layer; those transposes map poorly to the
ANE, and the cost is weight-dependent -- it shows up only on trained weights (which peak
~5x higher in magnitude than random init), which is why a random-weight benchmark hides
it. `bench_latency.py --real` compares both layouts on the trained checkpoint.

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

| encoder                  | latency  | hardware |
| ------------------------ | -------: | -------- |
| whisper.cpp Metal        | ~7.3 ms  | GPU      |
| ANEForge (channels-first)| ~10.5 ms | ANE      |
| whisper.cpp CoreML       | ~11.2 ms | ANE      |
| ANEForge (generic seq,d) | ~33.6 ms | ANE      |

On the ANE, the channels-first ANEForge encoder is faster than CoreML's own encoder
(~10.5 vs ~11.2 ms in-engine), which is the result that matters: CoreML is the only
other way to reach the ANE, and going direct through ANEForge now beats it, with no
CoreML dependency or conversion step and a trainable graph. The optimized Metal kernel
is still faster for tiny (~7.3 ms), so on latency alone the GPU wins; the ANE's case is
energy (~5x less than MPS). The generic `[seq, d]` ANEForge encoder is included to show
what the ANE-native layout buys (~3x).

These ANEForge numbers are also measured end to end inside whisper.cpp: the encoder is
wired in as a backend (mirroring the CoreML seam, in a fork), transcribes jfk.wav
correctly, and runs at the same ~10.5 ms in `whisper-bench`.

Reproduce the whisper.cpp side: build it with and without `-DWHISPER_COREML=1` (the
CoreML encoder needs a converted `ggml-tiny-encoder.mlmodelc`), then
`whisper-bench -m models/ggml-tiny.bin`.

## Notes

- Energy is the ANE's strongest result: ~30 mJ vs PyTorch-MPS's ~160 mJ (about 5x),
  idle-subtracted whole-package, with the fast encoder. Measured against eager MPS, not
  a tuned Metal kernel, but the ANE rail is low (~2.6 W active over a 9.5 ms encode).
- At seq 1500 `af.sdpa` decomposes to the same matmul/softmax as `af.mha`, because the
  native fused-attention layer is reliable only when the smaller attention axis is
  below 512. The channels-first encoder uses `einsum` attention, not `af.sdpa`.
- The fast encoder is wired into whisper.cpp end to end (a backend mirroring the CoreML
  seam, in a fork) and transcribes correctly, at the same ~10.5 ms in-engine.

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
