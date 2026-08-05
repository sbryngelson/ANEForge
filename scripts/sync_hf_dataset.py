#!/usr/bin/env python3
"""Mirror the committed roofline datapoints to the aneforge/ane-rooflines Hugging Face dataset.

Runs in CI on push to main (see .github/workflows/sync-hf-dataset.yml). Needs a secret HF_TOKEN
with write access to the aneforge org; if it is unset (e.g. on a fork), this is a no-op so the
workflow stays green. The dataset's README card is left untouched -- only the data files are synced.
"""
import glob
import os
import shutil
import sys
import tempfile

DATASET = "aneforge/ane-rooflines"


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set; skipping dataset sync.")
        return 0
    from huggingface_hub import upload_folder  # lazy: only needed when syncing

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    staging = tempfile.mkdtemp()
    shutil.copy(os.path.join(root, "bench/results/rooflines.json"), os.path.join(staging, "rooflines.json"))
    shutil.copy(os.path.join(root, "bench/results/ROOFLINES.md"), os.path.join(staging, "ROOFLINES.md"))
    os.makedirs(os.path.join(staging, "raw"))
    for p in sorted(glob.glob(os.path.join(root, "bench/results/rooflines/roofline-*.json"))):
        shutil.copy(p, os.path.join(staging, "raw", os.path.basename(p)))

    url = upload_folder(folder_path=staging, repo_id=DATASET, repo_type="dataset", token=token,
                        commit_message="Sync roofline datapoints from ANEForge main")
    print("synced:", url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
