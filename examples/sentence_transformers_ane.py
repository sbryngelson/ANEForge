"""Drop-in sentence-transformers SentenceTransformer running the encoder on the ANE. Run: python3 examples/sentence_transformers_ane.py"""
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
