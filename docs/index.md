# ANEForge - using the tool

ANEForge is a CoreML-free Python frontend for the Apple Neural Engine: it lowers
a tensor graph into one fused Espresso `e5rt` program and dispatches it to ANE
silicon from an ordinary user process, with no CoreML and no special entitlement.
These docs cover how to install, call, train, target, and extend the frontend.

## By task

### Get something running

- [Getting started](getting-started.md) - install, build, first program.
- [aneforge API](aneforge-api.md) - the graph -> compile -> run frontend reference.
- [Training on the ANE](training.md) - on-ANE autograd and the `Trainer` loop.
- [FAQ](faq.md) - common questions, gotchas, what to expect.
- [MIL primer](mil-primer.md) - writing MIL programs by hand.

### Run pretrained models

- [LLMs on the ANE](llm.md) - `af.load_llm` prefill + resident-KV-cache decode for
  Llama/Qwen/Mistral/Gemma/GPT-2/MoE, speculative decoding, and MoE from GGUF.
- [Model loaders](developer/models.md) - vision (ResNet / ViT / CLIP), sentence
  encoders and rerankers, GPT-2, and Whisper speech-to-text (both towers on the ANE).
- [ONNX import](onnx.md) - run an existing `.onnx` model on the engine via `af.load_onnx`.

### Use the engine well

- [Cross-chip deployment](cross-chip.md) - compiling and gating for other ANE
  families (M1-M5, 28 targets), `cross_compile_check`, `detect_family`, fp16 portability.
- [Dispatch backends](dispatch.md) - Path A vs e5rt vs MPSGraph vs CoreML, and which to use.
- [e5rt dispatch reference](e5rt-dispatch-reference.md) - the full e5rt path: call
  sequence, the `ane_e5rt_*` C ABI, multi-op / async / pipelining / IOSurface.

### Know what works

- [Capabilities](capabilities.md) - operator coverage, dtype matrix, known limits.
- [Op catalog](op-catalog.md) - every native MIL op x device (M1-M5), generated from
  the package's `_op_catalog.py` (the runtime `af.op_info` data); the exhaustive Y/~/N table.

### Contribute

- [Development](development.md) - building, testing, adding ops.
- [Glossary](glossary.md) - terminology used across docs and code.
- [Roadmap](roadmap.md) - next directions, open unknowns, and known bottlenecks.

## By question

| Question | Document |
| --- | --- |
| How do I install + run? | [getting-started](getting-started.md) |
| How do I use the Python frontend? | [aneforge-api](aneforge-api.md) |
| How do I train a model on the ANE? | [training](training.md) |
| How do I run an LLM, Whisper, or ONNX model? | [llm](llm.md), [model loaders](developer/models.md), [onnx](onnx.md) |
| Can I target / deploy to another chip (M1-M5)? | [cross-chip](cross-chip.md) |
| How do I estimate latency without the hardware? | [aneforge-api: cost estimation](aneforge-api.md#cost-estimation-measurement-free), [cross-chip](cross-chip.md) |
| How do I shrink weights (int4 / sparse)? | [aneforge-api: weight compression](aneforge-api.md#weight-compression) |
| What ops are supported? | [capabilities](capabilities.md), [op-catalog](op-catalog.md) |
| What's "Path A"? | [glossary](glossary.md), [dispatch](dispatch.md) |
| How do I write a MIL program? | [mil-primer](mil-primer.md) |
| Why fp16? | [faq](faq.md), [capabilities](capabilities.md) |
| Can I use this in production? | [faq](faq.md) |
| What macOS versions are supported? | [faq](faq.md) |
| How do I add a new operator? | [development](development.md) |
| Why does my call take 195 ms? | [dispatch](dispatch.md#choosing-a-path) |
