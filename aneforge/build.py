"""Build-on-demand for the e5rt dispatch dylib.

The package ships the dispatch shim SOURCE (links Apple's private frameworks, so it is
compiled per-machine), built on first ANE dispatch (or `python -m aneforge.build`) and
cached. ANEFORGE_NO_AUTOBUILD=1 requires an explicit build. See docs/developer/compile-pipeline.md.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

_DYLIB = "libane_e5rt_dispatch.dylib"


def _lib_dir() -> Path:
  return Path(__file__).resolve().parent / "_lib"


def _cache_dir() -> Path:
  """Per-version build cache, used when the installed package tree is read-only."""
  from . import __version__
  root = os.environ.get("ANEFORGE_CACHE")
  base = Path(root) if root else Path.home() / ".cache" / "aneforge"
  return base / __version__


def build_dylib(out_path: Path | None = None, *, quiet: bool = False) -> Path:
  """Compile the e5rt dispatch dylib from the packaged source; return its path."""
  if sys.platform != "darwin":
    raise RuntimeError(
      f"{_DYLIB} can only be built on Apple Silicon macOS (got {sys.platform}); "
      "aneforge dispatches to the Apple Neural Engine, which exists only there.")
  src = _lib_dir() / "ane_e5rt_dispatch.mm"
  if not src.exists():
    raise FileNotFoundError(f"packaged dispatch source missing: {src}")
  if out_path is None:
    # Build next to the source (matches build.sh and the canonical lookup); fall
    # back to a user cache dir if the installed package tree is read-only.
    out_path = _lib_dir() / _DYLIB
    if not os.access(_lib_dir(), os.W_OK):
      out_path = _cache_dir() / _DYLIB
  out_path.parent.mkdir(parents=True, exist_ok=True)
  if not quiet:
    print(f"aneforge: building {_DYLIB} (one-time, ~1s)...", file=sys.stderr)
  # Compile to a temp file in the target dir, then atomically move it into place so a
  # concurrent import never loads a half-written dylib.
  fd, tmp = tempfile.mkstemp(dir=str(out_path.parent), suffix=".dylib"); os.close(fd)
  cmd = ["xcrun", "clang++", "-O2", "-Wall", "-Wextra", "-dynamiclib", "-fPIC",
         "-fobjc-arc", "-framework", "Foundation", "-I", str(_lib_dir()),
         str(src), "-o", tmp]
  try:
    r = subprocess.run(cmd, capture_output=True, text=True)
  except FileNotFoundError as e:
    os.unlink(tmp)
    raise RuntimeError(
      "xcrun not found; install the Xcode command-line tools with "
      "`xcode-select --install`, then retry.") from e
  if r.returncode != 0:
    try: os.unlink(tmp)
    except OSError: pass
    raise RuntimeError(
      f"failed to build {_DYLIB} (clang++ rc={r.returncode}). Ensure the Xcode "
      f"command-line tools are installed (`xcode-select --install`).\n{r.stderr}")
  os.replace(tmp, out_path)
  if not quiet:
    print(f"aneforge: built {out_path}", file=sys.stderr)
  return out_path


def main() -> None:
  print(build_dylib())


if __name__ == "__main__":
  main()
