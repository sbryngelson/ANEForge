"""Bridge bundles with several weight files are merged into one (aneforge/_bridges/_netplist.py):
macOS 27.2 rejects a bundle with two or more (#286). Pure plist/bytes logic, off-device."""
import plistlib

import pytest

from aneforge._bridges import _netplist as g


def test_merge_weights_rebases_every_reference(tmp_path):
  src = tmp_path / "src"
  g.write_model("sdpa", src, sdpa_channels=6, sdpa_sequence=2, sdpa_dim=16, sdpa_scale=0.25)
  assert sorted(p.name for p in src.glob("weights.*")) == ["weights.0", "weights.1"]
  net, weights = g.merge_weights(src / "net.plist", sorted(src.glob("weights.*")), tmp_path / "out")
  network = plistlib.loads(net.read_bytes())["network_sdpa-1"]
  assert [w.name for w in weights] == ["weights.0"] and network["Weights"] == ["weights.0"]
  # weights.0 was empty, so the scale (fp16 0.25 = 0x3400) lands at offset 0 of the merged file
  assert (network["scale"]["ConstantIndex"], network["scale"]["ConstantByteOffset"]) == (0, 0)
  assert weights[0].read_bytes()[:2] == bytes.fromhex("0034") and len(weights[0].read_bytes()) % 64 == 0


def test_merge_weights_offsets_follow_file_order(tmp_path):
  pl = {"net": {"Weights": ["weights.0", "weights.1", "weights.2"],
                "a": {"ConstantIndex": 2, "ConstantByteOffset": 4},
                "u": {"Params": {"KernelIndex": 1, "KernelOffset": 0}}}}
  (tmp_path / "net.plist").write_bytes(plistlib.dumps(pl))
  for i, n in enumerate((100, 3, 8)): (tmp_path / f"weights.{i}").write_bytes(bytes([i + 1]) * n)
  # passed out of order: the base offsets come from the file index, not the argument order
  net, (w,) = g.merge_weights(tmp_path / "net.plist", [tmp_path / f"weights.{i}" for i in (2, 0, 1)], tmp_path / "out")
  out = plistlib.loads(net.read_bytes())["net"]
  assert out["u"]["Params"] == {"KernelIndex": 0, "KernelOffset": 128}       # 100 B padded to 128
  assert out["a"] == {"ConstantIndex": 0, "ConstantByteOffset": 192 + 4}     # 128 + 64, plus its own 4
  blob = w.read_bytes()
  assert blob[128:131] == b"\x02" * 3 and blob[192:200] == b"\x03" * 8 and len(blob) == 256


def test_invoke_retries_merged_only_after_the_bundle_check_fails(tmp_path, monkeypatch):
  calls = []

  def fake(invoker, net_plist, weights, **kw):
    calls.append(len(weights))
    if len(weights) > 1: raise RuntimeError('x failed: "verifyBundleAtPath: invalid model"')
    return {"status": "ok"}
  monkeypatch.setattr(g, "_invoke", fake)
  monkeypatch.setattr(g, "_MERGE_WEIGHTS", False)
  g.write_model("sdpa", tmp_path, sdpa_channels=6, sdpa_sequence=2, sdpa_dim=16)
  ws = sorted(tmp_path.glob("weights.*"))
  assert g.invoke_netplist("inv", tmp_path / "net.plist", weights=ws) == {"status": "ok"}
  assert calls == [2, 1] and g._MERGE_WEIGHTS
  g.invoke_netplist("inv", tmp_path / "net.plist", weights=ws)   # remembered: no second failed attempt
  assert calls == [2, 1, 1]
  # any other failure is not retried
  monkeypatch.setattr(g, "_MERGE_WEIGHTS", False)
  monkeypatch.setattr(g, "_invoke", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ANECCompile() FAILED")))
  with pytest.raises(RuntimeError, match="ANECCompile"): g.invoke_netplist("inv", tmp_path / "net.plist", weights=ws)
  assert not g._MERGE_WEIGHTS
