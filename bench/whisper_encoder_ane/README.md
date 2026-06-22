# Whisper encoder on the Apple Neural Engine

The Whisper audio encoder, compiled with ANEForge into one program and run on the
Apple Neural Engine (no CoreML), benchmarked against the same encoder in PyTorch (CPU
and Metal/MPS) and against whisper.cpp's own Metal and CoreML encoders.

The encoder is the fixed-shape half of Whisper (an 80-channel log-mel over a fixed
1500-frame context), so it compiles once and dispatches many times. The
autoregressive decoder, whose KV-cache length changes every token, is not covered
here.

## Result (whisper-tiny encoder, M-series)

| engine               | latency | energy / encode | fidelity      |
| -------------------- | ------: | --------------: | ------------- |
| ANE                  | 11.4 ms | ~32 mJ          | cosine 0.9998 |
| PyTorch-MPS (eager)  | 13.0 ms | ~126 mJ         | reference     |
| PyTorch CPU (fp32)   | 36.1 ms | --              | reference     |

On the ANE the encoder runs at about 11 ms per encode and uses about 3.8x less energy
than the PyTorch-MPS baseline, at cosine 0.9998 on real speech with an identical
transcript (see Fidelity). Latency is not the headline: an optimized GPU kernel is
faster than the ANE here (see Compared to whisper.cpp). The case for the ANE is energy.
The energy ratio is stable across runs; absolute milliJoules move with background
system load, and it is measured against PyTorch-MPS, not an optimized Metal kernel,
which would draw less and narrow the ratio.

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

## Compared to whisper.cpp

The PyTorch-MPS baseline above is an eager GPU path, not an optimized kernel.
whisper.cpp ships both an optimized Metal encoder and a CoreML encoder that reaches
the ANE, so it is the real point of comparison. Measured with its own `whisper-bench`
on the same tiny model, steady-state per encode:

| encoder            | latency  | hardware |
| ------------------ | -------: | -------- |
| whisper.cpp Metal  | ~7.3 ms  | GPU      |
| whisper.cpp CoreML | ~11.2 ms | ANE      |
| ANEForge direct    | ~11.4 ms | ANE      |

Two things follow. The optimized Metal kernel is the latency winner for tiny, so on
the ANE the case is energy, not speed. And ANEForge's direct path matches CoreML on
ANE latency: going direct does not beat CoreML on speed, it ties it. The direct path
wins elsewhere: no CoreML dependency or conversion step, the encoder is trainable, and
the cold cost is lower. A single cold encode in a fresh process is about 12 ms for
ANEForge, against about 36 ms for whisper.cpp's CoreML path (CoreML runtime plus
first-inference setup) and 14 ms for Metal; ANEForge pays its setup once, as a ~0.6 s
compile at load.

Reproduce the whisper.cpp side: build it with and without `-DWHISPER_COREML=1` (the
CoreML encoder needs a converted `ggml-tiny-encoder.mlmodelc`), then
`whisper-bench -m models/ggml-tiny.bin`.

## Notes

- The case for the ANE is energy. On its own rail it draws about 1.9 W against the
  PyTorch-MPS path's 8 W (idle-subtracted, whole package); an optimized Metal kernel
  draws less than eager MPS, so the gap to a tuned GPU path is smaller than 3.8x.
- At seq 1500 `af.sdpa` decomposes to the same matmul/softmax as `af.mha`, because the
  native fused-attention layer is reliable only when the smaller attention axis is
  below 512. The two rows are identical by construction.
- The comparison still missing is a real in-engine integration: wiring the ANEForge
  encoder into whisper.cpp's decode path and measuring word-error-rate end to end,
  which needs a whisper.cpp fork.

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
