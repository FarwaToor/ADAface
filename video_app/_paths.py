"""Import first in any script that needs the shared `common/` modules
(model, config, face_alignment, AdaFace) — makes them importable regardless
of the current working directory."""
import os
import sys

_COMMON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common")
if _COMMON_DIR not in sys.path:
    sys.path.append(_COMMON_DIR)
