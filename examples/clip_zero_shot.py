"""CLIP zero-shot image classification on the Apple Neural Engine (aneforge), running both the
Vision Transformer and causal Text Transformer encoders on the ANE (#179).
Run: python3 examples/clip_zero_shot.py"""
import sys, time

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af

NAME = "openai/clip-vit-base-patch32"


def main():
    _common.head("CLIP zero-shot classification on the Apple Neural Engine (aneforge)")
    print("config: openai/clip-vit-base-patch32 | ViT-B/32 vision + causal text encoder | 512-dim shared latent")

    import torch
    from transformers import CLIPModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NAME)
    hf = CLIPModel.from_pretrained(NAME).eval()

    print("\nloading CLIP via aneforge (compiling vision + text programs for ANE) ...", end="", flush=True)
    t0_load = time.perf_counter()
    clip = af.load_clip(NAME)
    dt_load = time.perf_counter() - t0_load
    print(f" done ({dt_load:.2f}s)")

    # Candidate labels
    labels = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a sports car",
        "a photo of an airplane",
        "a photo of a tropical beach",
    ]

    # Synthetic image (RGB pattern with distinctive colored block)
    img_array = np.zeros((1, 3, 224, 224), dtype=np.float32)
    img_array[:, 0, 50:150, 50:150] = 2.0   # Red channel
    img_array[:, 1, 50:150, 50:150] = 0.5   # Green channel
    img_array[:, 2, :, :] = -1.0             # Blue channel

    print(f"\nimage input: [1, 3, 224, 224] | {len(labels)} candidate text prompts")

    # 1. Vision encode on ANE
    t0 = time.perf_counter()
    ane_img_feat = clip.encode_image(img_array)
    dt_img = time.perf_counter() - t0

    # 2. Text encode on ANE
    t0 = time.perf_counter()
    ane_txt_feat = clip.encode_text(labels)
    dt_txt = time.perf_counter() - t0

    # 3. Classify (zero-shot scoring)
    t0 = time.perf_counter()
    ranked = clip.classify(img_array, labels)
    dt_cls = time.perf_counter() - t0

    # 4. PyTorch reference validation
    txt_inputs = tok(labels, padding="max_length", max_length=clip.St, return_tensors="pt")
    with torch.no_grad():
        hf_out = hf(input_ids=txt_inputs["input_ids"], pixel_values=torch.tensor(img_array))
        hf_img_norm = hf_out.image_embeds.numpy()
        hf_txt_norm = hf_out.text_embeds.numpy()
        hf_logits = hf_out.logits_per_image.numpy()[0]
        hf_exp = np.exp(hf_logits - np.max(hf_logits))
        hf_probs = hf_exp / np.sum(hf_exp)

    cos_img = float(ane_img_feat.ravel() @ hf_img_norm.ravel() / (np.linalg.norm(ane_img_feat) * np.linalg.norm(hf_img_norm)))
    cos_txts = [
        float(ane_txt_feat[i] @ hf_txt_norm[i] / (np.linalg.norm(ane_txt_feat[i]) * np.linalg.norm(hf_txt_norm[i])))
        for i in range(len(labels))
    ]
    min_cos_txt = min(cos_txts)

    print("\n" + "=" * 55)
    print("ZERO-SHOT CLASSIFICATION RANKING (ANE):")
    print("=" * 55)
    for rank, (label, prob) in enumerate(ranked, 1):
        idx = labels.index(label)
        hf_p = float(hf_probs[idx])
        print(f"  #{rank}: {label:<28} ANE={prob * 100:5.2f}% | HF={hf_p * 100:5.2f}% (diff={abs(prob - hf_p) * 100:.2f}%)")

    print("\nVALIDATION & BENCHMARKS:")
    print(f"  Vision embedding cosine vs HF fp32: {cos_img:.5f}")
    print(f"  Text embeddings min cosine vs HF:   {min_cos_txt:.5f}")
    print(f"  Vision latency on ANE:              {dt_img * 1e3:.2f} ms")
    print(f"  Text latency ({len(labels)} prompts):          {dt_txt * 1e3:.2f} ms ({dt_txt / len(labels) * 1e3:.2f} ms/prompt)")
    print(f"  Total classify latency:             {dt_cls * 1e3:.2f} ms")

    # Parity checks
    vision_ok = cos_img > 0.99
    text_ok = min_cos_txt > 0.99
    top1_match = ranked[0][0] == labels[int(np.argmax(hf_probs))]

    print("\n" + "=" * 55)
    print(f"Vision encoder validates (cos>0.99): {vision_ok}")
    print(f"Text encoder validates (cos>0.99):   {text_ok}")
    print(f"Top-1 classification matches HF:     {top1_match}")
    ok = vision_ok and text_ok and top1_match
    print("RESULT:", "PASS" if ok else "FAIL")
    print("=" * 55)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())