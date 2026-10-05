"""The submission fingerprint's main merge-base (bench/_machine.py, #290): it must be
taken against the canonical repo's main, not a fork's possibly-stale origin/main.
Throwaway git repos with hand-set remote refs; off-device, no network."""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
import _machine as m   # noqa: E402

CANON_HTTPS = "https://github.com/sbryngelson/ANEForge.git"
FORK_HTTPS = "https://github.com/someone/ANEForge.git"


class TestCanonicalRemote:
  """Picking the canonical remote out of `git remote -v` by URL."""

  def _v(self, *pairs):
    return "\n".join(f"{n}\t{u} ({k})" for n, u in pairs for k in ("fetch", "push"))

  def test_upstream_in_fork_checkout(self):
    assert m._canonical_remote(self._v(("origin", FORK_HTTPS), ("upstream", CANON_HTTPS))) == "upstream"

  def test_origin_in_direct_clone(self):
    assert m._canonical_remote(self._v(("origin", CANON_HTTPS))) == "origin"

  def test_matched_by_url_not_name(self):
    # 'upstream' points somewhere else; the canonical repo is under another name
    v = self._v(("origin", FORK_HTTPS), ("upstream", "https://github.com/other/ANEForge"),
                ("sb", "git@github.com:sbryngelson/ANEForge.git"))
    assert m._canonical_remote(v) == "sb"

  def test_url_spellings(self):
    for u in ("https://github.com/sbryngelson/ANEForge", "https://github.com/sbryngelson/ANEForge/",
              "https://github.com/SBryngelson/aneforge.git", "git@github.com:sbryngelson/ANEForge.git",
              "ssh://git@github.com/sbryngelson/ANEForge.git"):
      assert m._canonical_remote(self._v(("x", u))) == "x", u

  def test_lookalike_repos_do_not_match(self):
    for u in ("https://github.com/sbryngelson/ANEForge-pub.git", "https://github.com/notsbryngelson/ANEForge.git",
              FORK_HTTPS):
      assert m._canonical_remote(self._v(("x", u))) is None, u

  def test_no_remotes(self):
    assert m._canonical_remote("") is None
    assert m._canonical_remote(None) is None


def _g(repo, *args):
  return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def fork_checkout(tmp_path, monkeypatch):
  """A fork checkout whose origin/main is 3 commits behind the canonical main, with
  HEAD on canonical main. Remote refs are set directly, so nothing is fetched."""
  repo = tmp_path / "repo"
  repo.mkdir()
  _g(repo, "init", "-q", "-b", "main")
  _g(repo, "config", "user.email", "t@example.com")
  _g(repo, "config", "user.name", "t")
  shas = []
  for i in range(4):
    _g(repo, "commit", "-q", "--allow-empty", "-m", f"c{i}")
    shas.append(_g(repo, "rev-parse", "HEAD"))
  _g(repo, "remote", "add", "origin", FORK_HTTPS)
  _g(repo, "update-ref", "refs/remotes/origin/main", shas[0])     # stale fork main
  monkeypatch.setattr(m, "REPO", repo)
  return repo, shas


class TestGitInfoMainRef:
  def test_fork_without_canonical_remote_uses_origin(self, fork_checkout):
    # the old behavior, still the fallback: drift against the fork's own main
    _, shas = fork_checkout
    main = m._git_info()["main"]
    assert main["ref"] == "origin/main" and main["merge_base"] == shas[0]
    assert main["commits_ahead"] == 3 and main["canonical"] is False

  def test_canonical_remote_wins_over_stale_origin(self, fork_checkout):
    # THE BUG (#285): the run was on canonical main, but the table showed origin's base (+N)
    repo, shas = fork_checkout
    _g(repo, "remote", "add", "upstream", CANON_HTTPS)
    _g(repo, "update-ref", "refs/remotes/upstream/main", shas[3])
    main = m._git_info()["main"]
    assert main["ref"] == "upstream/main" and main["merge_base"] == shas[3]
    assert main["commits_ahead"] == 0 and main["canonical"] is True

  def test_canonical_remote_under_another_name(self, fork_checkout):
    repo, shas = fork_checkout
    _g(repo, "remote", "add", "upstream", "https://github.com/other/ANEForge.git")
    _g(repo, "update-ref", "refs/remotes/upstream/main", shas[1])
    _g(repo, "remote", "add", "sb", "git@github.com:sbryngelson/ANEForge.git")
    _g(repo, "update-ref", "refs/remotes/sb/main", shas[3])
    main = m._git_info()["main"]
    assert main["ref"] == "sb/main" and main["merge_base"] == shas[3] and main["canonical"] is True

  def test_canonical_remote_not_fetched_falls_through(self, fork_checkout):
    # remote added but never fetched: no sb/main ref, so the next candidate is used
    repo, shas = fork_checkout
    _g(repo, "remote", "add", "sb", CANON_HTTPS)
    main = m._git_info()["main"]
    assert main["ref"] == "origin/main" and main["canonical"] is False

  def test_no_remotes_uses_local_main(self, tmp_path, monkeypatch):
    repo = tmp_path / "solo"
    repo.mkdir()
    _g(repo, "init", "-q", "-b", "main")
    _g(repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "c0")
    monkeypatch.setattr(m, "REPO", repo)
    main = m._git_info()["main"]
    assert main["ref"] == "main" and main["canonical"] is False
