"""The suite's report is written as strict JSON (bench/_machine.py json_safe): a non-finite
float becomes null, not the bare NaN that jq and JSON.parse reject. Off-device."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
import _machine as m   # noqa: E402


def test_json_safe_writes_no_bare_nan():
  report = {"a": float("nan"), "b": [1.5, float("inf"), {"c": float("-inf"), "d": "NaN"}], "e": (2, None)}
  safe = m.json_safe(report)
  assert safe == {"a": None, "b": [1.5, None, {"c": None, "d": "NaN"}], "e": [2, None]}
  json.dumps(safe, allow_nan=False)   # raises on a non-finite float


def test_dumps_strict_nulls_what_a_default_handler_produces():
  class Scalar:   # stands in for a numpy scalar, which json.dumps hands to `default`
    def __init__(self, v): self.v = v
  text = m.dumps_strict({"x": Scalar(float("nan")), "y": Scalar(2.5), "z": float("inf")}, indent=2,
                        default=lambda o: o.v)
  assert json.loads(text) == {"x": None, "y": 2.5, "z": None} and "NaN" not in text and text.startswith("{\n  ")
