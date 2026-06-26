"""Import an ONNX model and run it on the ANE, validated against onnxruntime. With no argument
this exports torchvision ResNet-18 to ONNX as a self-contained demo; pass a path to import your
own classifier instead. Run: python3 examples/onnx_import.py [model.onnx]"""
import os
import sys
import tempfile

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af


def _demo_onnx(path):
    """Export torchvision ResNet-18 to `path` as the default demo model (random weights are fine -- we compare against onnxruntime, not ImageNet)."""
    import torch, torchvision
    m = torchvision.models.resnet18(weights=None).eval()
    torch.onnx.export(m, (torch.randn(1, 3, 224, 224),), path, opset_version=13,
                      do_constant_folding=True, input_names=["x"], output_names=["y"], dynamo=False)
    return path


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else _demo_onnx(os.path.join(tempfile.mkdtemp(), "resnet18.onnx"))

    inputs, _ = af.onnx_to_tensor(path)            # the uncompiled graph, to read the input shape
    net = af.load_onnx(path)                        # parse -> build aneforge graph -> compile
    print(f"imported {os.path.basename(path)}: {net.n_ops} ops fused into 1 ANE program")

    rng = np.random.default_rng(0)
    x = rng.standard_normal(inputs[0].shape).astype(np.float32)
    ane = np.asarray(net(x.astype(np.float16))).astype(np.float32).ravel()      # fp16 on the ANE

    import onnxruntime as ort
    sess = ort.InferenceSession(path)
    ref = np.asarray(sess.run(None, {sess.get_inputs()[0].name: x})[0]).astype(np.float32).ravel()

    cos = float(ane @ ref / (np.linalg.norm(ane) * np.linalg.norm(ref)))
    print(f"output cosine(ANE fp16, onnxruntime fp32) = {cos:.4f}")
    print(f"ANE top-5:        {ane.argsort()[-5:][::-1].tolist()}")
    print(f"onnxruntime top-5: {ref.argsort()[-5:][::-1].tolist()}")
    print(f"top-1 match: {int(ane.argmax()) == int(ref.argmax())}")


if __name__ == "__main__":
    sys.exit(main())
