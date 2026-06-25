"""torchvision ResNet-18 as one fused ANE program, validated vs fp32. Run: python3 examples/resnet18.py"""
import sys

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af


def main():
    clf = af.load_resnet18()
    print(f"ResNet-18: {clf.n_ops} ops fused into 1 ANE program")

    rng = np.random.default_rng(0)
    img = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
    ane = clf(img)[0]

    import torch, torchvision
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
    with torch.no_grad():
        ref = m(torch.from_numpy(img)).numpy()[0]

    cos = float(ane @ ref / (np.linalg.norm(ane) * np.linalg.norm(ref)))
    print(f"logit cosine(ANE, torchvision) = {cos:.4f}")
    print(f"ANE top-5: {ane.argsort()[-5:][::-1].tolist()}")
    print(f"ref top-5: {ref.argsort()[-5:][::-1].tolist()}")
    print(f"top-1 match: {int(ane.argmax()) == int(ref.argmax())}")


if __name__ == "__main__":
    sys.exit(main())
