"""Host-side plumbing that makes a RoBERTa/XLM-R reranker differ from a BERT one.

Off-device on purpose: the encoder graph is identical for both families, so what a regression would
break is the weight selection and the position indexing, and both are pure numpy. The end-to-end
check against `transformers` on real weights needs an ANE and a download; it lives in the PR body.
"""
import numpy as np
import pytest

from aneforge.models import (
  _DISTILBERT_KEYS,
  _encoder_layer_spec,
  _position_ids,
  _seqcls_activation,
  _seqcls_head,
)

BERT_SD = {
  "bert.embeddings.word_embeddings.weight": None,
  "bert.pooler.dense.weight": None, "bert.pooler.dense.bias": None,
  "classifier.weight": None, "classifier.bias": None,
}
# As shipped by BAAI/bge-reranker-base: no pooler, a two-layer classifier, and a token-type table
# that exists with exactly one row.
ROBERTA_SD = {
  "roberta.embeddings.word_embeddings.weight": None,
  "roberta.embeddings.token_type_embeddings.weight": None,
  "classifier.dense.weight": None, "classifier.dense.bias": None,
  "classifier.out_proj.weight": None, "classifier.out_proj.bias": None,
}
DISTILBERT_SD = {
  "distilbert.embeddings.word_embeddings.weight": None,
  "distilbert.transformer.layer.0.attention.q_lin.weight": None,
  "pre_classifier.weight": None, "pre_classifier.bias": None,
  "classifier.weight": None, "classifier.bias": None,
}


def test_bert_head_keys():
  assert _seqcls_head(BERT_SD, "bert") == (
    "bert.pooler.dense.weight", "bert.pooler.dense.bias", "classifier.weight", "classifier.bias")


def test_roberta_head_keys():  # RobertaClassificationHead: dense -> tanh -> out_proj, no pooler
  assert _seqcls_head(ROBERTA_SD, "roberta") == (
    "classifier.dense.weight", "classifier.dense.bias",
    "classifier.out_proj.weight", "classifier.out_proj.bias")


def test_distilbert_head_keys():
  assert _seqcls_head(DISTILBERT_SD, "distilbert", "distilbert") == (
    "pre_classifier.weight", "pre_classifier.bias", "classifier.weight", "classifier.bias")


def test_distilbert_encoder_layer_spec():
  prefix, keys = _encoder_layer_spec("distilbert")
  assert prefix == "transformer.layer.{i}."
  assert keys is _DISTILBERT_KEYS
  assert keys["Wq"] == "attention.q_lin.weight"
  assert keys["ln2w"] == "output_layer_norm.weight"


def test_bert_encoder_layer_spec_is_unchanged():
  prefix, keys = _encoder_layer_spec("bert")
  assert prefix == "encoder.layer.{i}."
  assert keys["Wq"] == "attention.self.query.weight"


def test_distilbert_head_uses_relu_and_bert_head_uses_tanh():
  values = np.array([-2.0, 0.0, 2.0])
  np.testing.assert_array_equal(_seqcls_activation(values, "distilbert"), [0.0, 0.0, 2.0])
  np.testing.assert_allclose(_seqcls_activation(values, "bert"), np.tanh(values))


def test_roberta_is_not_detected_by_a_missing_token_type_table():
  """The obvious discriminator does not work: XLM-R *has* token_type_embeddings, so keying on its
  absence would classify bge-reranker-base as BERT and then look for a pooler that is not there."""
  assert "roberta.embeddings.token_type_embeddings.weight" in ROBERTA_SD
  assert _seqcls_head(ROBERTA_SD, "roberta")[0] == "classifier.dense.weight"


def test_head_is_none_for_an_unsupported_model():
  assert _seqcls_head({"distilbert.embeddings.word_embeddings.weight": None}, "distilbert") is None
  assert _seqcls_head(BERT_SD, None) is None          # no prefix -> no pooler key to look for


def test_bert_positions_count_from_zero():
  ids = np.array([101, 2054, 2003, 102], dtype=np.int64)
  assert _position_ids(ids, None).tolist() == [0, 1, 2, 3]


def test_roberta_positions_are_offset_by_padding_idx_plus_one():
  """HF create_position_ids_from_input_ids. With pad_token_id=1 the first real token is position 2,
  which is why the table carries max_position_embeddings + 2 rows."""
  ids = np.array([0, 2367, 83, 2], dtype=np.int64)     # <s> ... </s>, no pads
  assert _position_ids(ids, 1).tolist() == [2, 3, 4, 5]


def test_roberta_positions_skip_pads():
  ids = np.array([0, 2367, 2, 1, 1], dtype=np.int64)   # two trailing <pad>
  assert _position_ids(ids, 1).tolist() == [2, 3, 4, 1, 1]


@pytest.mark.parametrize("pad_id", [0, 1, 3])
def test_roberta_positions_stay_inside_the_table(pad_id):
  """Every index must be addressable in a table of max_position + 2 rows, for any padding_idx."""
  ids = np.arange(100, 140, dtype=np.int64)
  pos = _position_ids(ids, pad_id)
  assert pos.min() >= 0 and pos.max() < len(ids) + pad_id + 1
