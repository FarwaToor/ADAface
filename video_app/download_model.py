#!/usr/bin/env python3
"""Downloads the extra model weights needed by video_processing.py:
the fine-tuned YOLOv8 face detector and dlib's 68-point landmark predictor.
Not required for fr.py alone."""
import bz2
import os
import urllib.request

import gdown

YOLO_URL = 'https://drive.google.com/uc?id=1IJZBcyMHGhzAi0G4aZLcqryqZSjPsps-'
YOLO_OUTPUT_PATH = 'yolov8m_200e.pt'

LANDMARK_PREDICTOR_URL = (
    'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
)
LANDMARK_PREDICTOR_OUTPUT_PATH = 'shape_predictor_68_face_landmarks.dat'


def download_yolov8_model(model_url=YOLO_URL, output_path=YOLO_OUTPUT_PATH):
    if os.path.exists(output_path):
        print(f"{output_path} already exists, skipping download.")
        return
    gdown.download(model_url, output_path, quiet=False)


def download_landmark_predictor(url=LANDMARK_PREDICTOR_URL, output_path=LANDMARK_PREDICTOR_OUTPUT_PATH):
    if os.path.exists(output_path):
        print(f"{output_path} already exists, skipping download.")
        return
    compressed_path = output_path + '.bz2'
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, compressed_path)
    with bz2.BZ2File(compressed_path) as fr, open(output_path, 'wb') as fw:
        fw.write(fr.read())
    os.remove(compressed_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    download_yolov8_model()
    download_landmark_predictor()
