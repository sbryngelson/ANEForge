# Pretrained models

Loaders that import pretrained weights and fuse the model into one ANE program.
This covers the vision, encoder, reranker, CLIP, GPT-2, and Whisper loaders in
`aneforge.models`. Decoder LLMs load via `af.load_llm` (see the [LLM guide](../llm.md))
and ONNX models via `af.load_onnx` (see [ONNX import](../onnx.md)).

::: aneforge.models
