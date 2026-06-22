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
| ANE (ANEForge)       | 5.9 ms  | ~31 mJ          | cosine 0.9998 |
| PyTorch-MPS (eager)  | 13.0 ms | ~156 mJ         | reference     |
| PyTorch CPU (fp32)   | 35 ms   | --              | reference     |

The encoder runs on the ANE at 5.9 ms per encode and about 5x less energy than the
PyTorch-MPS baseline, at cosine 0.9998 on real speech with an identical transcript (see
Fidelity). On the same ANE it is about 2x faster than CoreML's own encoder, and it also
beats the optimized Metal GPU kernel -- at every size from tiny to medium (see Compared
to whisper.cpp).

### How it gets there

A direct `[seq, d]` translation of the encoder runs at ~33 ms (3x slower than this, and
slower than the GPU). Two layout choices move it onto the ANE's strengths:

- **Channels-first throughout.** `build_cf` keeps every tensor in `[1, d_model, 1, S]`,
  the layout the ANE is built for: projections are 1x1 convolutions, the norm is
  `channel_layer_norm` (LayerNorm over channels, added to ANEForge for this), and there
  are no `[seq, d]` transposes. The generic `[seq, d]` layout (`build`) instead reshapes
  per layer; those transposes map poorly to the ANE, and the cost is weight-dependent --
  it appears only on trained weights (which peak ~5x higher than random init), so a
  random-weight benchmark hides it.
- **Query-tiled attention (kernel fission).** Each head's query axis is split into tiles
  (`q_tiles=3` for S=1500), so the score matrix is materialized as `[S, S/3]` tiles
  rather than the full `[S, S]` -- flash-attention's idea, but expressed as `einsum` so
  there is no transpose penalty. The smaller score tiles pipeline far better on the ANE:
  roughly 2x on attention, and exact (each query tile attends to all keys, so the
  transcript is unchanged -- verified, below).
- **int4 weights at scale.** `compress="int4"` streams quantized weights through the
  ANE's dequant path. It does nothing at tiny but buys ~13% at medium, whose MLP is
  weight-bandwidth-bound -- a win CoreML's ANE path does not get from quantization.

`bench_latency.py --real` compares the layouts on the trained checkpoint.

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
(`wer_proxy.py`) -- and that holds for the optimized encoders too: query-tiled at tiny
and small, and query-tiled + int4 at medium, all transcribe jfk.wav identically to the
reference. That is one clip, not a dataset-wide WER, but it shows the decoder absorbs
the gap on real speech. (Fidelity is robust to the weights; latency is not, as
noted above.)

## Compared to whisper.cpp, across sizes

whisper.cpp ships both an optimized Metal encoder and a CoreML encoder that reaches the
ANE -- the real points of comparison (CoreML is the only other way onto the ANE). The
ANEForge column is the channels-first, query-tiled encoder (int4 for medium); CoreML and
Metal are whisper.cpp's own, fp16, via its `whisper-bench`. All are the trained
checkpoint, per encode:

| model  | ANEForge | CoreML (ANE) | Metal (GPU) | vs CoreML | vs Metal |
| ------ | -------: | -----------: | ----------: | --------: | -------: |
| tiny   |  5.7 ms  |    11.2 ms   |    7.3 ms   |  **2.0x** |  1.3x    |
| base   | 12.2 ms  |    23.0 ms   |   13.5 ms   |  **1.9x** |  1.1x    |
| small  | 40.3 ms  |    77.2 ms   |   40.9 ms   |  **1.9x** |  1.0x    |
| medium |117.6 ms  |   236.0 ms   |  120.0 ms   |  **2.0x** |  1.0x    |

So going direct through ANEForge is about **2x faster than CoreML at every size**, and
also **beats or matches the Metal GPU** -- at ~5x less energy and cosine 0.999+. The
CoreML gap is structural: CoreML lays out attention reasonably but gets no speedup from
quantization on the ANE (int8 medium: 236 -> 235 ms, despite 4x smaller weights), while
ANEForge streams int4 through the dequant path. The Metal win comes from the query-tiling
-- without it the ANE only ties the GPU.

These numbers hold end to end inside whisper.cpp: the encoder is wired in as a backend
(mirroring the CoreML seam, in a fork) and transcribes jfk.wav correctly at the same
latency. Reproduce the whisper.cpp side by building with and without
`-DWHISPER_COREML=1` (CoreML needs a converted `ggml-<size>-encoder.mlmodelc`), then
`whisper-bench -m models/ggml-<size>.bin`.

## Notes

- Energy: ~31 mJ vs PyTorch-MPS's ~156 mJ (about 5x), idle-subtracted whole-package.
  Measured against eager MPS, not a tuned Metal kernel, but the ANE rail is low.
- At seq 1500 the native fused-attention layer (`af.sdpa`) is unreliable, so the encoder
  uses `einsum` attention; query-tiling recovers flash-attention's benefit without it.
- The fast encoder is wired into whisper.cpp end to end (a backend mirroring the CoreML
  seam, in a fork) and transcribes correctly, at the same latency in-engine.

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
