#!/bin/sh
# Community fuzz run: fuzz the compiler against the numpy reference on YOUR Mac's ANE and
# print a paste-ready report. Any Apple Silicon Mac works; chip diversity is the point.
#
#   bash scripts/fuzz.sh              # 15 minutes (default)
#   bash scripts/fuzz.sh 30           # 30 minutes
#
# If it reports findings, open a GitHub issue titled "fuzz: finding <fingerprint>" and paste
# the block below plus the finding-*.json file(s). Ten people hitting the same quirk is one
# issue: search for the fingerprint first.
set -e
cd "$(dirname "$0")/.."
MINUTES="${1:-15}"

echo "== ANEForge community fuzz run =="
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)
OS=$(sw_vers -productVersion 2>/dev/null || echo unknown)
REV=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
echo "chip:    $CHIP"
echo "macos:   $OS"
echo "commit:  $REV"
echo "budget:  ${MINUTES} min"
echo

ANEFORGE_DISABLE_COMPILE_BREAKER=1 python3 scripts/fuzz.py --minutes "$MINUTES" --out fuzz-report.json
STATUS=$?

echo
echo "---- paste-ready report ----"
echo "chip: $CHIP | macos: $OS | commit: $REV | budget: ${MINUTES}m"
python3 - <<'EOF'
import json
r = json.load(open("fuzz-report.json"))
print(f"graphs: {r['graphs']} | elapsed: {r['elapsed_s']}s | master seed: {r['master_seed']}")
if not r["findings"]:
  print("findings: none - this machine agrees with the reference on every generated graph")
for f in r["findings"]:
  print(f"finding {f['fingerprint']}: {f['fails'][0]['kind']} at opt={f['fails'][0]['opt']} "
        f"({f['nodes_after']} nodes) - attach finding-{f['fingerprint']}.json")
EOF
echo "----------------------------"
exit $STATUS
