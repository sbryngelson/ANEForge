#!/usr/bin/env python3
"""Machine fingerprint for the roofline suite: identifies WHICH Apple Silicon
machine produced a result, precisely enough to pool data across contributors.

The subtlety this solves: identical silicon ships in different thermal envelopes.
An M2 Pro in a fanless MacBook Air, a 16-inch MacBook Pro, and a Mac mini will
sustain different clocks and therefore different rooflines - so the fingerprint
carries the model IDENTIFIER (Mac14,7 vs Mac14,10 vs Mac14,12), not just the chip
string, and hashes the stable hardware fields into a short `hardware_hash` that
groups genuinely-identical machines while separating those three.

Fields split into two groups:
  - HARDWARE (fed into hardware_hash): model_identifier, chip, cpu_brand, core
    counts, gpu_cores, ram_gb, ane_rated_tops. Same machine -> same hash.
  - ENVIRONMENT (recorded, NOT hashed): model_name, macOS version/build,
    aneforge version, thermal state, sudo availability, timestamp. These vary run
    to run on the same machine, so they must not fragment the grouping key.

Every submission also gets a unique `run_id` so two contributors on the same
machine model never overwrite each other's committed JSON.

Run standalone to print this machine's fingerprint:
  PYTHONPATH=. python3 bench/_machine.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _int(key: str) -> int | None:
    v = _sysctl(key)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


def _sw_vers(flag: str) -> str | None:
    try:
        out = subprocess.run(["sw_vers", flag], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _gpu_cores() -> int | None:
    """Parse GPU core count from SPDisplaysDataType (best-effort; can be slow)."""
    try:
        out = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                             capture_output=True, text=True, timeout=20)
        m = re.search(r"Total Number of Cores:\s*(\d+)", out.stdout)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _model_name() -> str | None:
    try:
        out = subprocess.run(["system_profiler", "SPHardwareDataType"],
                             capture_output=True, text=True, timeout=20)
        m = re.search(r"Model Name:\s*(.+)", out.stdout)
        return m.group(1).strip() if m else None
    except Exception:
        return None


def _thermal_state() -> str:
    """pmset thermal/performance warning level, or 'nominal' when none recorded."""
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5)
        txt = out.stdout
        m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", txt)
        if m and m.group(1) != "100":
            return f"throttled(cpu_speed_limit={m.group(1)})"
        return "nominal"
    except Exception:
        return "unknown"


def _have_sudo() -> bool:
    try:
        return subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0
    except Exception:
        return False


def _aneforge_version() -> str | None:
    try:
        import aneforge as af
        return getattr(af, "__version__", None)
    except Exception:
        return None


def _git(*args) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO), *args],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _git_info() -> dict:
    """The repo commit the benchmark ran from AND how it relates to canonical main.

    A local checkout drifts: it can be a feature branch, sit behind main, or carry
    uncommitted edits - so results taken 'today' may not match main tomorrow. We
    pin all of it: the exact HEAD, dirty flag, and the main merge-base (the commit
    on main this code derives from) plus ahead/behind counts. main_ref prefers the
    canonical remote (origin/main) over a possibly-stale local main."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    info = {
        "commit": commit,
        "short": commit[:12] if commit else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }

    # relationship to canonical main (prefer origin/main; fall back to local main)
    main_ref = None
    for ref in ("origin/main", "main"):
        if _git("rev-parse", "--verify", ref):
            main_ref = ref
            break
    if main_ref:
        base = _git("merge-base", "HEAD", main_ref)
        counts = _git("rev-list", "--left-right", "--count", f"{main_ref}...HEAD")
        behind = ahead = None
        if counts and len(counts.split()) == 2:
            behind, ahead = (int(x) for x in counts.split())
        info["main"] = {
            "ref": main_ref,
            "merge_base": base,                    # the commit on main this corresponds to
            "merge_base_short": base[:12] if base else None,
            "commits_ahead": ahead,                # local commits not on main
            "commits_behind": behind,              # main commits not in this checkout
        }
    return info


def _power_source() -> dict:
    """AC vs battery + charge (a laptop on battery throttles clocks, shifting perf rooflines)."""
    try:
        txt = subprocess.run(["pmset", "-g", "ps"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return {"source": None, "is_laptop": None}
    is_laptop = "InternalBattery" in txt
    m = re.search(r"drawing from '([^']+)'", txt)
    info = {"source": ("ac" if (m and "AC" in m.group(1)) else "battery" if m else None),
            "is_laptop": is_laptop}
    if is_laptop:
        pct = re.search(r"(\d+)%", txt)
        info["battery_pct"] = int(pct.group(1)) if pct else None
        info["battery_state"] = ("discharging" if "discharging" in txt
                                 else "charging" if "charging" in txt
                                 else "charged" if "charged" in txt else None)
    return info


def _gpu_wired_limit_mb() -> int | None:
    """GPU-wired memory ceiling (iogpu.wired_limit_mb); 0 == system default (~=75% RAM)."""
    return _int("iogpu.wired_limit_mb")


def github_handle() -> str | None:
    """Best-effort GitHub handle from git config, resolved offline (no API).

    Only recoverable when the user commits with a GitHub noreply email
    (`ID+login@users.noreply.github.com` or `login@users.noreply.github.com`) -
    GitHub's default for web commits and a common local setting. A generic email
    (gmail etc.) cannot be mapped to a handle without the API, so this returns
    None and the caller falls back to an explicit --contributor. Deterministic:
    it reads git config, so the same machine yields the same answer in CI."""
    email = _git("config", "user.email") or ""
    m = re.match(r"^(?:\d+\+)?([A-Za-z0-9](?:[A-Za-z0-9-]{0,38})?)@users\.noreply\.github\.com$", email)
    return m.group(1) if m else None


# Published ANE peak throughput (TOPS, int8) by chip family. Left None where
# Apple has not published a figure - do NOT fabricate; a None is a fillable gap,
# a wrong number silently corrupts every "% of peak" downstream.
_RATED_TOPS = {
    "m1": 11.0, "m2": 15.8, "m3": 18.0, "m4": 38.0,
    "a14": 11.0, "a15": 15.8, "a16": 17.0, "a17": 35.0,
}


def _rated_tops(chip: str | None) -> float | None:
    if not chip:
        return None
    low = chip.lower()
    for key, val in _RATED_TOPS.items():
        if key in low:
            return val
    return None


def fingerprint() -> dict:
    """Full machine fingerprint: hardware identity + hardware_hash + environment + run_id."""
    chip = _sysctl("machdep.cpu.brand_string")
    memsize = _int("hw.memsize")
    hardware = {
        "model_identifier": _sysctl("hw.model"),           # Mac17,8 - the form-factor key
        "chip": chip,                                       # Apple M5 Pro
        "physical_cpu": _int("hw.physicalcpu"),
        "p_cores": _int("hw.perflevel0.physicalcpu"),
        "e_cores": _int("hw.perflevel1.physicalcpu"),
        "gpu_cores": _gpu_cores(),
        "memory_architecture": "unified",                   # CPU/GPU/ANE share one pool
        "ram_gb": (memsize or 0) // (1024 ** 3) or None,
        "ram_bytes": memsize,
        "ane_rated_tops": _rated_tops(chip),
    }
    hardware_hash = hashlib.sha256(
        json.dumps(hardware, sort_keys=True).encode()
    ).hexdigest()[:12]
    environment = {
        "model_name": _model_name(),                        # MacBook Pro / Mac mini / MacBook Air
        "macos_version": _sw_vers("-productVersion"),
        "macos_build": _sw_vers("-buildVersion"),
        "aneforge_version": _aneforge_version(),
        "aneforge_git": _git_info(),                        # exact code + relation to canonical main
        "power": _power_source(),                           # ac/battery: laptops throttle on battery
        "gpu_wired_limit_mb": _gpu_wired_limit_mb(),         # tunable GPU mem ceiling (0=default)
        "thermal_state": _thermal_state(),
        "have_sudo": _have_sudo(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return {
        "hardware": hardware,
        "hardware_hash": hardware_hash,
        "environment": environment,
        "run_id": uuid.uuid4().hex[:8],                     # unique per submission
    }


def chip_slug(fp: dict) -> str:
    """Filesystem-safe short chip tag, e.g. 'apple-m5-pro'."""
    chip = (fp["hardware"].get("chip") or "unknown").lower()
    return re.sub(r"[^a-z0-9]+", "-", chip).strip("-")


def result_filename(fp: dict, stem: str = "roofline") -> str:
    """Collision-free, groupable filename: stem-chip-model-hwhash-runid.json.

    hwhash groups identical machines; model_identifier separates form factors;
    run_id keeps every contributor's submission even on the same machine model."""
    model = re.sub(r"[^A-Za-z0-9]+", "_", fp["hardware"].get("model_identifier") or "unknown")
    return f"{stem}-{chip_slug(fp)}-{model}-{fp['hardware_hash']}-{fp['run_id']}.json"


def main():
    fp = fingerprint()
    print(json.dumps(fp, indent=2))
    print("\nresult filename ->", result_filename(fp))


if __name__ == "__main__":
    main()
