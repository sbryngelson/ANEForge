"""all-MiniLM-L6-v2 sentence encoder on the ANE with a tiny semantic search, validated vs fp32. Run: python3 examples/sentence_embeddings.py"""
import sys

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af


def main():
    embed = af.load("sentence-transformers/all-MiniLM-L6-v2", int8=False)

    corpus = [
        "The Apple Neural Engine accelerates neural networks at low power.",
        "Cats are independent animals that enjoy sleeping in the sun.",
        "Transformers process whole sequences in a single forward pass.",
        "The recipe calls for two cups of flour and a pinch of salt.",
        "Convolutional networks excel at image classification on the ANE.",
    ]
    docs = embed(corpus)                       # [N, D] on the ANE (L2-norm on-device)

    # Verify on-device L2Norm: cosine vs the transformers fp32 reference.
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        name = "sentence-transformers/all-MiniLM-L6-v2"
        tok = AutoTokenizer.from_pretrained(name)
        ref_model = AutoModel.from_pretrained(name).eval()
        with torch.no_grad():
            enc = tok(corpus, padding=True, truncation=True, return_tensors="pt")
            out = ref_model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1)
            ref = torch.nn.functional.normalize(pooled, dim=-1).numpy()
        cos = (docs * ref).sum(-1) / (np.linalg.norm(docs, axis=-1) * np.linalg.norm(ref, axis=-1))
        print(f"on-device L2Norm vs transformers reference: cosine mean {cos.mean():.4f} "
              f"min {cos.min():.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"(reference cosine check skipped: {type(e).__name__}: {e})")

    for query in ["How do I run machine learning efficiently on Apple hardware?",
                  "What do house pets like to do?"]:
        q = embed(query)[0]
        sims = docs @ q                        # cosine (vectors are normalised)
        order = np.argsort(-sims)
        print(f"\nQ: {query}")
        for rank, i in enumerate(order[:3]):
            print(f"  {rank+1}. ({sims[i]:.3f}) {corpus[i]}")


if __name__ == "__main__":
    sys.exit(main())
