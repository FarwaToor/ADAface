#!/usr/bin/env python3
"""Shared configuration for the AdaFace recognition core."""
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent

ADAFACE_REPO_DIR = BASE_DIR / "AdaFace"
ADA_MODEL_PATH = str(ADAFACE_REPO_DIR / "pretrained" / "adaface_ir101_webface12m.ckpt")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
