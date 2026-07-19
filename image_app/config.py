#!/usr/bin/env python3
"""Configuration for the image enrollment/search app. Override via environment
variables or a .env file (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
EMBEDDING_DIR = BASE_DIR / "embeddings"
SQLITE_DB_PATH = BASE_DIR / "person_data.db"

for _d in (UPLOAD_DIR, EMBEDDING_DIR):
    _d.mkdir(parents=True, exist_ok=True)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
