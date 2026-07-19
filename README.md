# ADAface

Face recognition built on [AdaFace](https://github.com/mk-minchul/AdaFace) (IR-101,
trained on WebFace12M). Two applications share the same recognition core:

- **`image_app/`** — a Flask web app to enroll face photos and search for a match
  (Redis for embeddings, SQLite for person records).
- **`video_app/`** — CLI tools to identify a person across face crops (`fr.py`), and
  optionally to extract those crops from a video in the first place
  (`video_processing.py`, via YOLOv8 person detection + DeepSort tracking).

## How it was tested

Both apps were run end-to-end against the real `adaface_ir101_webface12m.ckpt`
checkpoint: enrolling photos through `/enroll`, searching with `/search`, listing
enrolled people, and deleting a record, all via `image_app/app.py`'s Flask routes.
Along the way this fixed several bugs that made the original code non-runnable:

- The face-alignment model (`common/face_alignment`) was hardcoded to `device='cuda:0'`
  at import time, so it crashed immediately on any machine without an NVIDIA GPU.
  It now auto-detects CUDA and falls back to CPU.
- `image_app` imported `face_alignment` and `AdaFace` (the recognition model), but
  those directories only existed under the video app — `image_app` could not run
  at all as committed. Both are now shared from `common/`.
- The Flask app (previously `test.py`, a misleading name for the actual entry point)
  referenced a `delete_confirmation.html` template that doesn't exist in `templates/`.
- Every path (embeddings dir, uploads dir, SQLite file, Redis host, reference image,
  model checkpoint) was hardcoded to a personal `/home/farwa/...` path. All are now
  relative to the project or configurable via `.env`.
- `requirements.txt` didn't exist; dependencies (including `opencv-python`, silently
  required by the alignment code) are now pinned in one place.

## Tech stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Face recognition | AdaFace (IR-101 backbone), PyTorch |
| Face detection / alignment | MTCNN |
| Web app | Flask, Werkzeug |
| Embedding storage | Redis |
| Person records | SQLite |
| Video pipeline (optional) | YOLOv8 (ultralytics), DeepSort, dlib |

## Project structure

```
ADAface/
├── common/                    # shared recognition core (used by both apps)
│   ├── face_alignment/        # MTCNN face detection + alignment
│   ├── model.py                # loads AdaFace, extracts embeddings, computes similarity
│   ├── model_config.py         # checkpoint path, device (CPU/CUDA auto-detect)
│   └── AdaFace/                 # cloned by setup_env.py — not committed (see below)
├── image_app/                  # Flask enrollment/search web app
│   ├── app.py                   # entry point: routes for /enroll, /search, /delete_person, ...
│   ├── cli.py                   # command-line face search (no server needed)
│   ├── inspect_db.py            # dump the SQLite person table
│   ├── migrate_to_sqlite.py     # one-off Redis → SQLite migration helper
│   ├── config.py                # upload/embedding dirs, Redis host (reads .env)
│   ├── templates/, static/      # web UI
│   └── uploads/, embeddings/, person_data.db   # runtime data (git-ignored)
├── video_app/                   # video-based recognition
│   ├── fr.py                     # match a reference photo against a folder of face crops
│   ├── video_processing.py       # extract face crops from a video (YOLOv8 + DeepSort + dlib)
│   ├── download_model.py         # downloads YOLOv8 weights + dlib landmark predictor
│   └── requirements-video.txt    # extra deps for video_processing.py only
├── setup_env.py                 # clones AdaFace + downloads the recognition checkpoint
└── requirements.txt
```

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
# CPU-only machine? Install torch from the CPU index first:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu

python setup_env.py   # clones AdaFace and downloads the ~1.5GB IR-101 checkpoint
```

You'll also need a local [Redis](https://redis.io/docs/install/) instance for
`image_app` (`redis-server`, or `docker run -p 6379:6379 redis`).

## Run

**Image enrollment / search web app:**

```bash
cd image_app
cp .env.example .env   # only needed if Redis isn't on localhost:6379
python app.py
```

Open <http://localhost:4996> — enroll a photo, then search with another to find the
closest match. Or use the CLI directly: `python cli.py path/to/photo.jpg`.

**Video-based recognition:**

```bash
cd video_app
pip install -r requirements-video.txt   # only needed for video_processing.py
python download_model.py                 # YOLOv8 weights + dlib landmark predictor
python video_processing.py path/to/video.mp4          # extracts output_faces/
python fr.py path/to/reference_photo.jpg               # matches against output_faces/
```

`dlib` has no prebuilt wheel on every platform/Python version — it may need CMake and
a C++ build toolchain to compile from source. `fr.py` alone (matching against an
existing folder of face crops) doesn't need `dlib`, only `video_processing.py` does.

## Notes

- The AdaFace checkpoint (~1.5GB) and the cloned `AdaFace` architecture repo are not
  committed — `setup_env.py` fetches both. Enrolled photos, embeddings, and the
  SQLite database are runtime data and are also git-ignored.
- You may see a harmless NumPy deprecation warning when the MTCNN `.npy` weights load
  (they predate NumPy 2.x's stricter pickle format checks) — it doesn't affect results.
