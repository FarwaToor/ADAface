#!/usr/bin/env python3
"""One-time setup: clones the upstream AdaFace repo (for the model architecture
code) and downloads the pretrained IR-101 checkpoint into common/AdaFace/."""
import os
import subprocess
import sys

import gdown

COMMON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "common")
ADAFACE_DIR = os.path.join(COMMON_DIR, "AdaFace")
CHECKPOINT_URL = "https://drive.google.com/uc?id=1dswnavflETcnAuplZj1IOKKP0eM8ITgT"
CHECKPOINT_PATH = os.path.join(ADAFACE_DIR, "pretrained", "adaface_ir101_webface12m.ckpt")


def clone_repo():
    if os.path.exists(ADAFACE_DIR):
        print(f"{ADAFACE_DIR} already exists, skipping clone.")
        return
    subprocess.run(
        ["git", "clone", "https://github.com/mk-minchul/AdaFace", ADAFACE_DIR],
        check=True,
    )


def download_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        print(f"{CHECKPOINT_PATH} already exists, skipping download.")
        return
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    gdown.download(CHECKPOINT_URL, CHECKPOINT_PATH, quiet=False)


if __name__ == '__main__':
    clone_repo()
    download_checkpoint()
    print("Setup complete.")
    sys.exit(0)
