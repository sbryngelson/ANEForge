import numpy as np
import pytest
from _helpers import requires_ane
from aneforge import _blob


@pytest.fixture
def mkdir(tmp_path):
  """A factory for fresh build dirs that are auto-removed with the pytest tmp_path (no /tmp leak)."""
  n = [0]
  def _new():
    n[0] += 1; d = tmp_path / f"d{n[0]}"; d.mkdir(); return str(d)
  return _new


def _dequant_lut4(packed: bytes, lut: bytes, out: int, inn: int) -> np.ndarray:
  """Mirror of the on-device constexpr_lut_to_dense for test assertions."""
  codebook = np.frombuffer(lut, dtype=np.float16).astype(np.float32)
  raw = np.frombuffer(packed, dtype=np.uint8)
  lo, hi = raw & 0x0F, (raw >> 4) & 0x0F
  idx = np.empty(out * inn, dtype=np.uint8)
  idx[0::2], idx[1::2] = lo, hi
  return codebook[idx].reshape(out, inn)


def test_palettize_lut4_roundtrip_is_close():
  rng = np.random.default_rng(0)
  W = rng.standard_normal((32, 64)).astype(np.float32)
  packed, lut = _blob.palettize_lut4(W)
  assert len(lut) == 16 * 2                      # 16 fp16 centroids
  assert len(packed) == 32 * 64 // 2             # two 4-bit indices per byte
  recon = _dequant_lut4(packed, lut, 32, 64)
  rel = np.linalg.norm(recon - W) / np.linalg.norm(W)
  assert rel < 0.25                              # 16 levels on gaussian data


def test_palettize_lut4_is_deterministic():
  W = np.random.default_rng(1).standard_normal((16, 16)).astype(np.float32)
  a = _blob.palettize_lut4(W)
  b = _blob.palettize_lut4(W)
  assert a == b                                  # no RNG: byte-stable output


# Task 2 - compress= mode plumbing in _Emitter
from aneforge import _compile


def _tiny_matmul_graph():
  import aneforge as af
  x = af.input((1, 64))
  W = np.random.default_rng(2).standard_normal((64, 32)).astype(np.float32)
  return x @ W


def test_emitter_accepts_compress_mode():
  em = _compile._Emitter(int8=False, compress=None)
  assert em.compress is None
  em8 = _compile._Emitter(int8=False, compress="int8")
  assert em8.compress == "int8"


@requires_ane
def test_compress_none_matches_int8_false_mil():
  """compress=None must be byte-identical to the historical int8=False path."""
  import aneforge as af
  a = af.compile(_tiny_matmul_graph(), build_dir=None)       # historical default
  b = af.compile(_tiny_matmul_graph(), compress=None)        # new knob, off
  assert a.n_ops == b.n_ops


# Task 3 - int4-LUT weight branch + accuracy gate in _Emitter.weight
def test_weight_int4_falls_back_when_inaccurate():
  """A weight 16 levels can't represent within budget falls back, never errors."""
  em = _compile._Emitter(int8=False, compress="int4", compress_atol=1e-4)
  # 256 widely-spread values -> 16 centroids blow the budget
  W = (np.arange(256, dtype=np.float32) * 7.0).reshape(1, 256)
  name = em.weight("w", W, allow_int8=True, allow_int4=True)
  mil = "\n".join(em.lines)
  assert name == "w"
  assert "constexpr_lut_to_dense" not in mil          # int4 rejected by the gate
  assert "constexpr_affine_dequantize" in mil         # falls back to int8 (not fp16)


def test_int4_gate_not_bypassed_by_fp16_norm_overflow():
  """Regression: fp16 weight whose self-norm overflows fp16 must still be gated (rel-error in fp32)."""
  # fp16-representable values but W.dot(W)~2.7e8 overflows fp16; 16 LUT levels can't fit -> reject int4
  W = (np.arange(256, dtype=np.float32) * 7.0).reshape(1, 256).astype(np.float16)
  em = _compile._Emitter(int8=False, compress="int4", compress_atol=1e-3)
  em.weight("w", W, allow_int8=True, allow_int4=True)
  mil = "\n".join(em.lines)
  assert "constexpr_lut_to_dense" not in mil           # not silently accepted via inf denom
  assert "constexpr_affine_dequantize" in mil          # correctly falls back to int8


def test_weight_int4_falls_to_fp16_when_int8_disallowed():
  em = _compile._Emitter(int8=False, compress="int4", compress_atol=1e-4)
  W = (np.arange(256, dtype=np.float32) * 7.0).reshape(1, 256)
  em.weight("w", W, allow_int8=False, allow_int4=True)
  mil = "\n".join(em.lines)
  assert "constexpr_lut_to_dense" not in mil and "constexpr_affine_dequantize" not in mil
  assert "const(" in mil                              # plain fp16


def test_weight_sparse_allzero_falls_back():
  em = _compile._Emitter(int8=False, compress="sparse")
  W = np.zeros((8, 16), dtype=np.float32)
  em.weight("w", W, allow_int8=True, allow_sparse=True)
  assert "constexpr_sparse_to_dense" not in "\n".join(em.lines)


def test_weight_int4_used_when_accurate():
  em = _compile._Emitter(int8=False, compress="int4", compress_atol=0.5)
  W = np.random.default_rng(3).standard_normal((32, 64)).astype(np.float32)
  em.weight("w", W, allow_int8=True, allow_int4=True)
  assert "constexpr_lut_to_dense" in "\n".join(em.lines)


# Task 4 - on-device tests: int4-LUT runs on the ANE + weights.bin is smaller


@requires_ane
def test_int4_matmul_runs_on_ane_and_is_close():
  import aneforge as af
  rng = np.random.default_rng(4)
  W = rng.standard_normal((256, 128)).astype(np.float32)        # [IN, OUT]
  x = rng.standard_normal((1, 256)).astype(np.float32)
  ref = (x @ W).astype(np.float32)
  net = af.compile(af.input((1, 256)) @ W, compress="int4", compress_atol=0.5)
  got = net(x)
  cos = float((got.ravel() @ ref.ravel()) /
              (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-12))
  assert cos >= 0.99
  net.release()


@requires_ane
def test_int4_weights_are_smaller(mkdir):
  """The int4 program's weights.bin is markedly smaller than the fp16 one."""
  import aneforge as af
  from pathlib import Path
  W = np.random.default_rng(5).standard_normal((512, 256)).astype(np.float32)   # [IN, OUT]
  d16, d4 = mkdir(), mkdir()
  af.compile(af.input((1, 512)) @ W, compress=None, build_dir=d16).release()
  af.compile(af.input((1, 512)) @ W, compress="int4", compress_atol=0.5, build_dir=d4).release()
  s16 = Path(d16, "weights.bin").stat().st_size
  s4 = Path(d4, "weights.bin").stat().st_size
  assert s4 < s16 * 0.4                          # ~4x smaller (indices) + tiny codebook


@requires_ane
def test_compress_none_is_byte_identical_weights(mkdir):
  """compress=None must produce the SAME weights.bin as the historical default."""
  import aneforge as af
  from pathlib import Path
  rng = np.random.default_rng(6)
  W = rng.standard_normal((128, 64)).astype(np.float32)        # [IN, OUT]
  da, db = mkdir(), mkdir()
  af.compile(af.input((1, 128)) @ W, build_dir=da).release()                 # historical
  af.compile(af.input((1, 128)) @ W, compress=None, build_dir=db).release()  # explicit off
  assert Path(da, "weights.bin").read_bytes() == Path(db, "weights.bin").read_bytes()
  assert Path(da, "model.mil").read_text() == Path(db, "model.mil").read_text()


@requires_ane
def test_int4_conv_runs_on_ane_and_is_close(mkdir):
  import aneforge as af
  from pathlib import Path
  rng = np.random.default_rng(7)
  W = rng.standard_normal((16, 8, 3, 3)).astype(np.float32)     # flat inner = 8*3*3 = 72 (even)
  x = rng.standard_normal((1, 8, 16, 16)).astype(np.float32)
  base = af.compile(af.conv(af.input((1, 8, 16, 16)), W, pad=1), compress=None)
  ref = base(x); base.release()
  d = mkdir()
  net = af.compile(af.conv(af.input((1, 8, 16, 16)), W, pad=1),
                   compress="int4", compress_atol=0.6, build_dir=d)
  got = net(x); net.release()
  cos = float((got.ravel() @ ref.ravel()) /
              (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-12))
  assert cos >= 0.98
  # ensure it ACTUALLY used int4, not the silent fp16 fallback:
  assert b"constexpr_lut_to_dense" in Path(d, "model.mil").read_bytes()


# Task 8 - sparse weight streaming (constexpr_sparse_to_dense)
def test_sparsify_roundtrip():
  W = np.random.default_rng(8).standard_normal((32, 64)).astype(np.float32)
  W[np.abs(W) < 0.5] = 0.0                        # introduce real zeros
  mask, vals = _blob.sparsify(W)
  nnz = int(np.count_nonzero(W))
  assert len(vals) == nnz * 2                     # fp16 nonzeros
  assert len(mask) == (W.size + 7) // 8           # 1 bit per element
  # LSB-first scatter reconstructs the original (fp16 rounding only)
  bits = np.unpackbits(np.frombuffer(mask, dtype=np.uint8), bitorder="little")[: W.size].astype(bool)
  recon = np.zeros(W.size, dtype=np.float16)
  recon[bits] = np.frombuffer(vals, dtype=np.float16)
  # nonzero values are stored as fp16; compare against fp16-quantized W
  W_fp16 = W.astype(np.float16).astype(np.float32)
  assert np.array_equal(recon.reshape(W.shape).astype(np.float32), W_fp16)


def test_weight_sparse_used_when_sparse_enough():
  em = _compile._Emitter(int8=False, compress="sparse")
  W = np.random.default_rng(10).standard_normal((32, 64)).astype(np.float32)
  W[np.abs(W) < 1.0] = 0.0                        # ~68% zeros
  em.weight("w", W, allow_int8=True, allow_sparse=True)
  assert "constexpr_sparse_to_dense" in "\n".join(em.lines)


def test_weight_sparse_falls_back_when_dense():
  em = _compile._Emitter(int8=False, compress="sparse")
  W = np.random.default_rng(11).standard_normal((32, 64)).astype(np.float32)  # ~0% zeros
  name = em.weight("w", W, allow_int8=True, allow_sparse=True)
  mil = "\n".join(em.lines)
  assert name == "w"
  assert "constexpr_sparse_to_dense" not in mil   # not sparse enough -> fall back
  assert "const(" in mil


@requires_ane
def test_sparse_matmul_runs_on_ane_and_is_close(mkdir):
  import aneforge as af
  from pathlib import Path
  rng = np.random.default_rng(12)
  W = rng.standard_normal((256, 128)).astype(np.float32)       # [IN, OUT]
  W[np.abs(W) < 1.0] = 0.0                                     # ~68% zeros
  x = rng.standard_normal((1, 256)).astype(np.float32)
  ref = (x @ W).astype(np.float32)
  d = mkdir()
  net = af.compile(af.input((1, 256)) @ W, compress="sparse", build_dir=d)
  got = net(x)
  cos = float((got.ravel() @ ref.ravel()) /
              (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-12))
  assert cos >= 0.99                                          # sparse is lossless -> ~1.0
  assert b"constexpr_sparse_to_dense" in Path(d, "model.mil").read_bytes()
  net.release()


# Feature 1 - compress="auto": per-weight best-correct-smallest encoding
# precedence: sparse (if sparse) -> int4 (if accurate) -> int8 -> fp16
def test_auto_picks_sparse_on_sparse_weight():
  em = _compile._Emitter(int8=False, compress="auto")
  W = np.random.default_rng(20).standard_normal((32, 64)).astype(np.float32)
  W[np.abs(W) < 1.0] = 0.0                        # ~68% zeros
  em.weight("w", W, allow_int8=True, allow_int4=True, allow_sparse=True)
  mil = "\n".join(em.lines)
  assert "constexpr_sparse_to_dense" in mil       # sparse preferred when applicable
  assert "constexpr_lut_to_dense" not in mil


def test_auto_picks_int4_on_dense_accurate_weight():
  em = _compile._Emitter(int8=False, compress="auto", compress_atol=0.5)
  W = np.random.default_rng(21).standard_normal((32, 64)).astype(np.float32)  # dense, no zeros
  em.weight("w", W, allow_int8=True, allow_int4=True, allow_sparse=True)
  mil = "\n".join(em.lines)
  assert "constexpr_sparse_to_dense" not in mil   # not sparse-enough
  assert "constexpr_lut_to_dense" in mil          # int4 within budget


def test_auto_falls_to_int8_then_fp16():
  # int4-unfit weight: 256 widely-spread values -> 16 centroids blow a tight budget
  W = (np.arange(256, dtype=np.float32) * 7.0).reshape(1, 256)
  em = _compile._Emitter(int8=False, compress="auto", compress_atol=1e-4)
  em.weight("w", W, allow_int8=True, allow_int4=True, allow_sparse=True)
  mil = "\n".join(em.lines)
  assert "constexpr_lut_to_dense" not in mil
  assert "constexpr_sparse_to_dense" not in mil
  assert "constexpr_affine_dequantize" in mil     # int8 fallback
  # with int8 disallowed -> fp16
  em2 = _compile._Emitter(int8=False, compress="auto", compress_atol=1e-4)
  em2.weight("w", W, allow_int8=False, allow_int4=True, allow_sparse=True)
  mil2 = "\n".join(em2.lines)
  assert "constexpr_affine_dequantize" not in mil2
  assert "constexpr_lut_to_dense" not in mil2
  assert "const(" in mil2                          # plain fp16


@requires_ane
def test_auto_none_byte_identical(mkdir):
  """compress="auto" picking fp16 does not regress; compress=None stays byte-identical to default."""
  import aneforge as af
  from pathlib import Path
  rng = np.random.default_rng(22)
  # rms_norm gamma: forced to fp16 (allow_int8=False, no int4/sparse)
  g = rng.standard_normal((16,)).astype(np.float32)
  x = rng.standard_normal((1, 16)).astype(np.float32)
  base = af.compile(af.input((1, 16)).rms_norm(g), compress=None)
  ref = base(x); base.release()
  net = af.compile(af.input((1, 16)).rms_norm(g), compress="auto")
  got = net(x); net.release()
  cos = float((got.ravel() @ ref.ravel()) /
              (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-12))
  assert cos >= 0.999                              # fp16 weight either way -> identical

  # byte-identity of compress=None vs historical default
  W = rng.standard_normal((128, 64)).astype(np.float32)
  da, db = mkdir(), mkdir()
  af.compile(af.input((1, 128)) @ W, build_dir=da).release()
  af.compile(af.input((1, 128)) @ W, compress=None, build_dir=db).release()
  assert Path(da, "weights.bin").read_bytes() == Path(db, "weights.bin").read_bytes()


@requires_ane
def test_auto_matmul_runs_on_ane():
  import aneforge as af
  rng = np.random.default_rng(23)
  W = rng.standard_normal((256, 128)).astype(np.float32)       # [IN, OUT]
  W[np.abs(W) < 1.0] = 0.0                                     # ~68% zeros -> auto picks sparse
  x = rng.standard_normal((1, 256)).astype(np.float32)
  ref = (x @ W).astype(np.float32)
  net = af.compile(af.input((1, 256)) @ W, compress="auto")
  got = net(x)
  cos = float((got.ravel() @ ref.ravel()) /
              (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-12))
  assert cos >= 0.99                                          # sparse is lossless
  net.release()


# Feature - family-aware compress="auto": only natively-streaming encodings are auto candidates
def test_native_streams_table():
  from aneforge import _targets as TG
  # M1: only int4-LUT + sparse stream
  assert TG.native_streams(TG.Family.A13) == frozenset({"int4", "sparse"})
  # A14 (M2): int4/int8/sparse stream, blockwise folds
  assert TG.native_streams(TG.Family.A14) == frozenset({"int4", "int8", "sparse"})
  # A16 (M5): the broad set streams
  assert TG.native_streams(TG.Family.A16) == frozenset({"int4", "int8", "sparse", "blockwise"})


def _sparse_int4able_weight():
  """A weight both >=50% zeros (sparse-eligible) and int4-representable."""
  W = np.random.default_rng(30).standard_normal((32, 64)).astype(np.float32)
  W[np.abs(W) < 1.0] = 0.0                        # ~68% zeros
  return W


def test_auto_on_h13_streams_sparse():
  # sparse streams on h13, so auto keeps sparse-before-int4 precedence; filter bites on int8/blockwise
  em = _compile._Emitter(int8=False, compress="auto", compress_atol=0.5, family=2)
  em.weight("w", _sparse_int4able_weight(), allow_int8=True, allow_int4=True, allow_sparse=True)
  mil = "\n".join(em.lines)
  assert "constexpr_sparse_to_dense" in mil       # a native h13 stream, preferred
  assert "constexpr_lut_to_dense" not in mil


def test_auto_on_h13_int4_reject_falls_to_fp16_not_int8():
  # int4-unfit weight: on h13 int8 would fold, so auto must fall to fp16
  W = (np.arange(256, dtype=np.float32) * 7.0).reshape(1, 256)
  em = _compile._Emitter(int8=False, compress="auto", compress_atol=1e-4, family=2)
  em.weight("w", W, allow_int8=True, allow_int4=True, allow_sparse=True)
  mil = "\n".join(em.lines)
  assert "constexpr_lut_to_dense" not in mil
  assert "constexpr_affine_dequantize" not in mil  # int8 folds on h13 -> skipped
  assert "const(" in mil                           # plain fp16


def test_auto_on_a16_keeps_full_precedence():
  # family 5 (M5) streams everything -> sparse still preferred (unchanged behavior)
  em = _compile._Emitter(int8=False, compress="auto", compress_atol=0.5, family=5)
  em.weight("w", _sparse_int4able_weight(), allow_int8=True, allow_int4=True, allow_sparse=True)
  mil = "\n".join(em.lines)
  assert "constexpr_sparse_to_dense" in mil
  assert "constexpr_lut_to_dense" not in mil


def test_explicit_modes_not_filtered_by_family():
  # single-mode knobs are the user's call: compress="sparse" on h13 still emits sparse
  em = _compile._Emitter(int8=False, compress="sparse", family=2)
  em.weight("w", _sparse_int4able_weight(), allow_int8=True, allow_int4=True, allow_sparse=True)
  assert "constexpr_sparse_to_dense" in "\n".join(em.lines)


@requires_ane
def test_auto_target_h13_compiles_native_stream(mkdir):
  """End-to-end: compile(compress='auto', target='h13') -> sparse-eligible weight comes out sparse, runs."""
  import aneforge as af
  from pathlib import Path
  rng = np.random.default_rng(31)
  W = rng.standard_normal((256, 128)).astype(np.float32)
  W[np.abs(W) < 1.0] = 0.0                                     # ~68% zeros
  x = rng.standard_normal((1, 256)).astype(np.float32)
  d = mkdir()
  net = af.compile(af.input((1, 256)) @ W, compress="auto", compress_atol=0.5,
                   target="h13", build_dir=d)
  mil = Path(d, "model.mil").read_bytes()
  assert b"constexpr_sparse_to_dense" in mil
  assert b"constexpr_affine_dequantize" not in mil             # int8 folds on h13
  ref = (x @ W).astype(np.float32)
  got = net(x)
  cos = float((got.ravel() @ ref.ravel()) /
              (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-12))
  assert cos >= 0.97                              # sparse is exact on the kept values
  net.release()


# Feature 2 - conv-sparse (allow_sparse on conv); 4-D mask unverified on device
@requires_ane
def test_sparse_conv_runs_on_ane(mkdir):
  import aneforge as af
  from pathlib import Path
  rng = np.random.default_rng(24)
  W = rng.standard_normal((16, 8, 3, 3)).astype(np.float32)    # flat inner = 72 (even)
  W[np.abs(W) < 1.0] = 0.0                                     # ~68% zeros
  x = rng.standard_normal((1, 8, 16, 16)).astype(np.float32)
  base = af.compile(af.conv(af.input((1, 8, 16, 16)), W, pad=1), compress=None)
  ref = base(x); base.release()
  d = mkdir()
  net = af.compile(af.conv(af.input((1, 8, 16, 16)), W, pad=1),
                   compress="sparse", build_dir=d)
  got = net(x); net.release()
  cos = float((got.ravel() @ ref.ravel()) /
              (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-12))
  assert cos >= 0.98
  assert b"constexpr_sparse_to_dense" in Path(d, "model.mil").read_bytes()


# Feature 3 - blockwise-affine compression (constexpr_blockwise_shift_scale)
def test_quantize_blockwise_roundtrip():
  rng = np.random.default_rng(30)
  W = rng.standard_normal((32, 64)).astype(np.float32)
  data, scale, nblocks = _blob.quantize_blockwise(W, 16)
  assert nblocks == 64 // 16
  q = np.frombuffer(data, dtype=np.int8).astype(np.float32).reshape(32, nblocks, 16)
  sc = np.frombuffer(scale, dtype=np.float16).astype(np.float32).reshape(32, nblocks)
  recon = (q * sc[:, :, None]).reshape(32, 64)
  rel = np.linalg.norm(recon - W) / np.linalg.norm(W)
  assert rel < 0.02                              # per-block int8 is tight


def test_quantize_blockwise_indivisible_block_size():
  """block_size is clamped to a divisor of IN (falls toward per-tensor if needed)."""
  W = np.random.default_rng(31).standard_normal((8, 30)).astype(np.float32)  # 30 not /32
  data, scale, nblocks = _blob.quantize_blockwise(W, 32)
  assert 30 % nblocks == 0                        # nblocks divides IN
  assert len(np.frombuffer(scale, np.float16)) == 8 * nblocks


def test_weight_blockwise_used_when_accurate():
  em = _compile._Emitter(int8=False, compress="blockwise", compress_atol=0.1, block_size=16)
  W = np.random.default_rng(32).standard_normal((32, 64)).astype(np.float32)
  name = em.weight("w", W, allow_int8=True)
  mil = "\n".join(em.lines)
  assert name == "w"
  assert "constexpr_blockwise_shift_scale" in mil


def test_weight_blockwise_falls_back_when_disallowed():
  """blockwise needs allow_int8; conv weights (allow_int8=False) fall to fp16, never error."""
  em = _compile._Emitter(int8=False, compress="blockwise")
  W = np.random.default_rng(33).standard_normal((16, 32)).astype(np.float32)
  name = em.weight("w", W, allow_int8=False)
  mil = "\n".join(em.lines)
  assert name == "w"
  assert "constexpr_blockwise_shift_scale" not in mil
  assert "const(" in mil                          # plain fp16


@requires_ane
def test_blockwise_matmul_runs_on_ane_and_is_close(mkdir):
  import aneforge as af
  from pathlib import Path
  rng = np.random.default_rng(34)
  W = rng.standard_normal((256, 128)).astype(np.float32)       # [IN, OUT]
  x = rng.standard_normal((1, 256)).astype(np.float32)
  ref = (x @ W).astype(np.float32)
  d = mkdir()
  net = af.compile(af.input((1, 256)) @ W, compress="blockwise",
                   block_size=32, compress_atol=0.5, build_dir=d)
  got = net(x)
  cos = float((got.ravel() @ ref.ravel()) /
              (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-12))
  assert cos >= 0.99                              # per-block int8 -> very close
  assert b"constexpr_blockwise_shift_scale" in Path(d, "model.mil").read_bytes()
  net.release()


@requires_ane
def test_blockwise_weights_are_smaller(mkdir):
  """The blockwise program's weights.bin is smaller than fp16 (int8 data + tiny scales)."""
  import aneforge as af
  from pathlib import Path
  W = np.random.default_rng(35).standard_normal((512, 256)).astype(np.float32)   # [IN, OUT]
  d16, db = mkdir(), mkdir()
  af.compile(af.input((1, 512)) @ W, compress=None, build_dir=d16).release()
  af.compile(af.input((1, 512)) @ W, compress="blockwise", block_size=32,
             compress_atol=0.5, build_dir=db).release()
  s16 = Path(d16, "weights.bin").stat().st_size
  sb = Path(db, "weights.bin").stat().st_size
  assert sb < s16 * 0.6                           # ~2x smaller (int8 + per-block scales)


@requires_ane
def test_blockwise_compress_none_unaffected(mkdir):
  """Adding compress='blockwise' must not perturb the compress=None default."""
  import aneforge as af
  from pathlib import Path
  W = np.random.default_rng(36).standard_normal((128, 64)).astype(np.float32)
  da, db = mkdir(), mkdir()
  af.compile(af.input((1, 128)) @ W, build_dir=da).release()
  af.compile(af.input((1, 128)) @ W, compress=None, build_dir=db).release()
  assert Path(da, "weights.bin").read_bytes() == Path(db, "weights.bin").read_bytes()
  assert Path(da, "model.mil").read_text() == Path(db, "model.mil").read_text()


# CompressionFallbackWarning

import warnings
import aneforge as af


def test_compress_int4_fallback_warns():
  """compress='int4' that rejects weights emits CompressionFallbackWarning."""
  rng = np.random.default_rng(42)
  # random weights are hard to compress to 16 LUT levels: high rel error -> rejected
  W = (rng.standard_normal((128, 128)) * 0.5).astype(np.float32)
  em = _compile._Emitter(int8=False, compress="int4", compress_atol=1e-6)
  em.weight("w", W.astype(np.float16), allow_int8=True, allow_int4=True)
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _compile._compression_fallback_signal(em)
    hits = [x for x in w if issubclass(x.category, af.CompressionFallbackWarning)]
    assert len(hits) == 1
    msg = str(hits[0].message)
    assert "compress='int4'" in msg
    assert "1/1" in msg
    assert "int4: 1" in msg
    assert "applied int8" in msg


def test_compress_int4_accepted_no_warning():
  """compress='int4' that accepts all weights emits no warning."""
  # constant weights compress perfectly to LUT
  W = np.full((64, 64), 0.5, dtype=np.float16)
  em = _compile._Emitter(int8=False, compress="int4", compress_atol=0.05)
  em.weight("w", W, allow_int8=True, allow_int4=True)
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _compile._compression_fallback_signal(em)
    assert not any(issubclass(x.category, af.CompressionFallbackWarning) for x in w)


def test_compress_none_no_warning():
  """compress=None emits no CompressionFallbackWarning."""
  rng = np.random.default_rng(44)
  W = (rng.standard_normal((64, 64)) * 0.5).astype(np.float16)
  em = _compile._Emitter(int8=False, compress=None)
  em.weight("w", W, allow_int8=True)
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _compile._compression_fallback_signal(em)
    assert not any(issubclass(x.category, af.CompressionFallbackWarning) for x in w)


def test_compress_warning_is_filterable():
  """CompressionFallbackWarning can be silenced with filterwarnings."""
  rng = np.random.default_rng(45)
  W = (rng.standard_normal((128, 128)) * 0.5).astype(np.float32)
  em = _compile._Emitter(int8=False, compress="int4", compress_atol=1e-6)
  em.weight("w", W.astype(np.float16), allow_int8=True, allow_int4=True)
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    warnings.filterwarnings("ignore", category=af.CompressionFallbackWarning)
    _compile._compression_fallback_signal(em)
    assert not any(issubclass(x.category, af.CompressionFallbackWarning) for x in w)


def test_compress_mixed_accept_reject_counts():
  """Warning reports correct counts when some weights accept and some reject int4."""
  rng = np.random.default_rng(46)
  # constant weight -> accepted by int4
  W_good = np.full((64, 64), 0.5, dtype=np.float16)
  # random weight -> rejected by int4
  W_bad = (rng.standard_normal((64, 64)) * 0.5).astype(np.float32).astype(np.float16)
  em = _compile._Emitter(int8=False, compress="int4", compress_atol=1e-6)
  em.weight("good", W_good, allow_int8=True, allow_int4=True)
  em.weight("bad", W_bad, allow_int8=True, allow_int4=True)
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _compile._compression_fallback_signal(em)
    hits = [x for x in w if issubclass(x.category, af.CompressionFallbackWarning)]
    assert len(hits) == 1
    msg = str(hits[0].message)
    assert "1/2" in msg
    assert "int4: 1" in msg
