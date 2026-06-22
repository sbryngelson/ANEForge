#!/bin/sh
# End-to-end C++ demo: export the encoder bundle from Python, build the standalone
# runner against the dispatch dylib, and run it on the ANE (no Python at run time).
# Run from the repo root:  sh bench/whisper_encoder_ane/run_cpp.sh
set -e
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
dylib="$repo/aneforge/_lib/libane_e5rt_dispatch.dylib"
out="/tmp/whisper_enc_bundle"

[ -f "$dylib" ] || sh "$repo/aneforge/_lib/build.sh"

echo "== exporting bundle + test vectors (Python) =="
PYTHONPATH="$repo" python3 "$here/export_bundle.py" --out "$out" >/dev/null
echo "   done -> $out"

echo "== building C++ runner =="
xcrun clang++ -O2 -std=c++17 "$here/whisper_ane_run.cpp" "$dylib" -o /tmp/whisper_ane_run

echo "== running on the ANE (no Python) =="
python3 - "$out" <<'PY'
import json, subprocess, sys
io = sys.argv[1] + "/io"
m = json.load(open(io + "/manifest.json"))
(i0, n0), (i1, n1) = m["inputs"]
on, onn = m["output"]
subprocess.run(["/tmp/whisper_ane_run", m["build_dir"],
                i0, str(n0), f"{io}/{i0}.f16", i1, str(n1), f"{io}/{i1}.f16",
                on, str(onn), f"{io}/ref.f32"], check=True)
PY
