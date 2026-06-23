"""Drop-in sentence-transformers on the Apple Neural Engine.

`aneforge.sentence_transformers.SentenceTransformer` mirrors the `.encode` surface
of the sentence-transformers package, but runs the encoder on the ANE. It reads the
model's own pooling config, so a mean-pooled model (MiniLM, E5) and a cls-pooled
model (BGE, GTE) both produce the right vectors, matching the reference to cosine
~1.0 at a fraction of the GPU's energy.

    pip install "aneforge[models]"
    python3 examples/sentence_transformers_ane.py
"""
import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np

from aneforge.sentence_transformers import SentenceTransformer


def main():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print(f"  pooling mode read from config: {model.pooling_mode}")

    corpus = [
        "The Apple Neural Engine accelerates neural networks at low power.",
        "Cats are independent animals that enjoy sleeping in the sun.",
        "Transformers process whole sequences in a single forward pass.",
    ]
    query = "What hardware runs networks efficiently?"

    docs = model.encode(corpus)                    # [N, D] on the ANE
    q = model.encode(query)                        # a single string -> [D]
    scores = docs @ q
    best = int(np.argmax(scores))

    print(f"  query : {query}")
    print(f"  match : {corpus[best]}   (score {scores[best]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
