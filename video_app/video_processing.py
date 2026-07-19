#!/usr/bin/env python3
"""Extract the sharpest, most-frontal face crop per tracked person from a video.

Detects and tracks people with YOLOv8 + DeepSort, filters to frontal faces
with a dlib landmark predictor, and keeps the least-blurry crop per track ID.
Feed the resulting `output_faces/` folder into fr.py to identify who's in it.

Requires extra dependencies not needed by the rest of the project:
`ultralytics`, `deep_sort_realtime`, `dlib` (see README — dlib needs a C++
build toolchain and does not have a prebuilt wheel on every platform), plus
the YOLOv8 weights and dlib's 68-point landmark predictor (download_model.py).
"""
import argparse
import os

import cv2
import dlib
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def is_frontal_face(landmarks, min_landmarks=5):
    """Check if the face is frontal based on the number of key landmarks detected."""
    return len(landmarks.parts()) >= min_landmarks


def image_sharpness(image):
    """Calculate the sharpness of an image using the variance of the Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def process_video(yolo_model, predictor, video_path, output_dir='output_faces', confidence_threshold=0.7):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Failed to open video file. Check the file path.")

    os.makedirs(output_dir, exist_ok=True)

    # Initialize DeepSort tracker with adjusted parameters
    tracker = DeepSort(max_age=200, nn_budget=100, max_iou_distance=0.7)

    # Dictionary to keep track of the best face for each unique ID
    best_faces = {}

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Detect faces using YOLOv8 with the confidence threshold
        results = yolo_model.track(source=frame, imgsz=640, conf=confidence_threshold)

        # Prepare detections for DeepSort
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for box, score, class_id in zip(boxes, scores, class_ids):
                if int(class_id) != 0:  # Assuming face class ID is 0
                    continue

                x1, y1, x2, y2 = map(int, box[:4])
                detections.append([[x1, y1, x2 - x1, y2 - y1], score, int(class_id)])

        # Update DeepSort tracker with detections
        tracks = tracker.update_tracks(detections, frame=frame)

        # Loop over the tracks and assign IDs
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue  # Only consider tracks that have been updated in the current frame

            track_id = track.track_id
            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

            # Extract face from frame
            face_image = frame[ymin:ymax, xmin:xmax]

            # Detect landmarks for the extracted face
            dlib_rect = dlib.rectangle(xmin, ymin, xmax, ymax)
            landmarks = predictor(frame, dlib_rect)

            # Check if the face is frontal and has at least 5 landmarks
            if is_frontal_face(landmarks):
                sharpness = image_sharpness(face_image)

                # Store the clearest image for each unique ID
                if track_id not in best_faces or sharpness > best_faces[track_id]["sharpness"]:
                    best_faces[track_id] = {"image": face_image, "sharpness": sharpness}

    cap.release()

    # Save the clearest face for each unique ID
    for track_id, data in best_faces.items():
        cv2.imwrite(os.path.join(output_dir, f"clearest_face_ID_{track_id}.jpg"), data["image"])

    print(f"Saved {len(best_faces)} face crop(s) to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract the sharpest frontal face crop per tracked person from a video."
    )
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output-dir", default="output_faces/", help="Where to save extracted face crops")
    parser.add_argument("--yolo-weights", default="yolov8m_200e.pt", help="Path to YOLOv8 weights")
    parser.add_argument(
        "--landmark-predictor", default="shape_predictor_68_face_landmarks.dat",
        help="Path to dlib's 68-point face landmark predictor",
    )
    parser.add_argument("--confidence", type=float, default=0.7, help="Detection confidence threshold")
    args = parser.parse_args()

    yolo_model = YOLO(args.yolo_weights)
    face_predictor = dlib.shape_predictor(args.landmark_predictor)
    process_video(yolo_model, face_predictor, args.video_path, args.output_dir, args.confidence)
