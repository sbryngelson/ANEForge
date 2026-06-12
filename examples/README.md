# aneforge examples

Runnable demos of the aneforge frontend - one example per file, named for what it
shows. Run any of them from the repo root:

```sh
python3 examples/<name>.py
```

Each script has a module docstring explaining what it shows. `_common.py` holds the
only shared scaffolding (env setup, error metrics, conditioned test matrices); the
demo logic itself stays in each file so every example is self-contained.

**Start here**
- `demo.py` - the README hero: a small char transformer trains end to end on the
  ANE (forward + backward + Adam), then generates text from it one character per
  on-engine forward pass. This is the script recorded for the README animation
  (`docs/assets/demo.tape`).
- `fluid_vorticity.py` - the README fluid showpiece: the word "ANEForge" is painted as a
  passive dye and wound into thin glowing filaments by a 2-D incompressible Navier-Stokes
  flow (pseudo-spectral), every FFT in the 2,200-step loop on the ANE. Writes the animated
  `docs/assets/fluid_vorticity.png` (a full-colour APNG).
- `quickstart.py` - the clean API end to end: a CNN and a transformer encoder block,
  each fused into ONE ANE program, fp16 + int8.

**Pretrained models**
- `resnet18.py` - torchvision ResNet-18 (ImageNet) as one fused ANE program.
- `vit.py` - ViT-B/16, the full encoder forward.
- `sentence_embeddings.py` - all-MiniLM-L6-v2 + a tiny semantic search.
- `superres_espcn.py` - ESPCN super-resolution (sub-pixel conv / pixel-shuffle).

**Training fully on the ANE** (forward + backward + Adam, K steps unrolled into one
program per dispatch - no per-step host loop, and optimizer state stays RESIDENT
on-device across dispatches; via `af.UnrolledTrainer`)
- `train_mnist_mlp.py` - MNIST MLP.
- `train_mnist_cnn.py` - MNIST CNN (trainable conv from primitives).
- `train_transformer.py` - transformer block (attention differentiable on-engine);
  full-batch, so data + lr + state are all resident and the loop is just
  `prog.execute()` - the host feeds nothing per dispatch.
- `train_transformer_prenorm.py` - pre-norm transformer block (LayerNorm before
  attention and the MLP); the gradient flows through `layer_norm` and the norm's own
  affine trains too.
- `train_llama_block.py` - LLaMA-style block (RMSNorm pre-norm + SwiGLU FFN);
  exercises the `rms_norm` and `silu` backward rules with trainable RMSNorm gains.
- `train_cifar_cnn.py` - a real CNN trained from scratch on CIFAR-10 (conv->GroupNorm->
  ReLU->pool stack from primitives), forward + backward + Adam all on the engine, to
  ~71% test accuracy; compared against a PyTorch model of the same topology.
- `train_charlm.py` - a 4-layer causal LLaMA-style char language model trained end to
  end on the engine (token/positional embeddings, RMSNorm + SwiGLU, next-token loss),
  then generated from.
- `train_charlm_corpus.py` - the same model trained on a corpus and shown to generalize:
  held-out next-character accuracy well above the unigram baseline.
- `train_charlm_deep.py` - a 16-layer char LM via a layer-streamed (gradient-
  checkpointed) compile, so depth is not bounded by compile size.

**LLM inference (decode on the engine)**
- `llama_block_causal.py` - a LLaMA decoder block running native CAUSAL attention
  (`af.sdpa(is_causal=True)`) on the ANE: the core GPT/LLaMA inference compute.
- `gpt_generate_ane.py` - end-to-end autoregressive GPT-style generation (prefill,
  KV-cache, per-step decode, greedy sampling) natively on the ANE.
- `gpt_multilayer_resident.py` - multi-layer GPT decode with every layer's KV-cache
  kept resident on the ANE across steps via `share_buffer` (zero host round-trip).

**Weight compression**
- `compress_weights.py` - int4-LUT / sparse / int8 streaming: accuracy x size x latency.

**Linear algebra** (LAPACK problem families on the engine; envelopes in docs/api/math.md)
- `solve_linear_systems.py` - conjugate gradient, K iterations as ONE fused program.
- `factorize.py` - QR / Cholesky / LU as fixed recurrences + pivoted LU (argmax pivot).
- `eigenvalues_svd.py` - full symmetric eig (unrolled + host-looped Jacobi), full SVD,
  generalized eig (sygv), nonsymmetric eig (geev), top-k SVD of a large matrix.
- `fft.py` - staged Cooley-Tukey as dense-DFT matmuls (sub-quadratic MACs).

**Applied math**
- `poisson_spectral.py` - spectral Poisson solver; each 2-D FFT is ONE fused program (`fft2`).
- `heat_equation.py` - 2D heat equation evolved over many timesteps.
- `spectral_analysis.py` - FFT-class spectral analysis of a real 1-D signal.
- `nbody.py` - gravitational force + integration step.
- `paired_fp16.py` - paired-fp16 (compensated) extended precision, no fp32 anywhere.

**Native hardware layers** (Path-A layer kinds Apple's public MIL/CoreML pipeline
never emits; each runs as a netplist-bridge graph cut, like `af.sdpa`)
- `sdpa.py` - the native fused-attention layer.
- `native_ranking.py` - sort / argmax / topk.
- `native_norms.py` - l2_norm / minmax_norm / lrn / scaled_elementwise.
- `native_geometry.py` - cross_product / cross_correlation / cost_volume / fps / radius_search.
- `native_pixel_ops.py` - pixel_shuffle/unshuffle, space<->channel, space<->batch, views.
- `pointcloud.py` - the geometry primitives composed into a PointNet++-style step.

**Stable Diffusion** (real diffusers weights; each component fuses into one ANE program)
- `sd_unet.py` - the full UNet2DConditionModel.
- `sd_vae.py` - the AutoencoderKL decoder.
- `sd15.py` - end-to-end SD-1.5 (host scheduler + CFG; shows the fp16 CFG-cancellation limit).

**Optimizer**
- `autotune.py` - `af.tune` measured speedups on real models, correctness preserved.

**Guided tour** (`demos/`)
- `demos/` - a sequenced set of small, single-topic programs that mirror the ANE guide's
  flow (the machine -> reaching it -> performance -> workloads -> practice): the execution
  model and dispatch floor, fp16 numerics, compile-without-CoreML, the roofline, batching
  and residency levers, and the capability surface. Each stands alone; see
  [`demos/README.md`](demos/README.md) for the full index.

Other files and directories:
- `benchmarks/` - throughput / dispatch benchmarks (encoder serving, ANE-vs-GPU baseline,
  persistent-worker dispatch), not how-to demos.
- `make_mnist_subset.py` - one-time data prep that produced the committed
  `data/mnist_subset.npz` the training demos load (kept for provenance; needs network).

The op-by-op smoke test lives in `tests/op_smoketest.py`.
