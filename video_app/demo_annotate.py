#!/usr/bin/env python3
"""Produce a short, shareable demo clip: detects and tracks faces in a video
(YOLOv8 + DeepSort) and overlays a live AdaFace similarity score against a
reference photo on every tracked face, frame by frame."""
import _paths  # noqa: F401
import argparse
import os
import tempfile

import cv2
import torch
import torch.nn.functional as F
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from face_alignment import align
from model import load_pretrained_model, to_input


def compute_reference_embedding(model, reference_path):
    aligned = align.get_aligned_face(reference_path)
    if aligned is None:
        raise ValueError(f"Could not detect a face in reference image {reference_path}")
    tensor = to_input(aligned)
    with torch.no_grad():
        feature, _ = model(tensor)
    return feature


def main():
    parser = argparse.ArgumentParser(description="Render an annotated face-recognition demo clip.")
    parser.add_argument("video_path", help="Input video")
    parser.add_argument("reference_image", help="Photo to compare every tracked face against")
    parser.add_argument("--output", default="demo_output.mp4", help="Output video path")
    parser.add_argument("--yolo-weights", default="yolov8m_200e.pt")
    parser.add_argument("--max-frames", type=int, default=240, help="Keep the clip short (default: 240 frames)")
    parser.add_argument("--match-threshold", type=float, default=0.3, help="Score above which a box is drawn green")
    parser.add_argument("--start-frame", type=int, default=0, help="Skip to this frame before recording")
    args = parser.parse_args()

    print("Loading AdaFace model...")
    model = load_pretrained_model('ir_101')
    reference_embedding = compute_reference_embedding(model, args.reference_image)

    print(f"Loading YOLOv8 weights from {args.yolo_weights}...")
    yolo_model = YOLO(args.yolo_weights)
    tracker = DeepSort(max_age=30, nn_budget=100, max_iou_distance=0.7)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {args.video_path}")
    if args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    fps = cap.get(cv2.CAP_PROP_FPS) or 12
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    tmp_dir = tempfile.mkdtemp()

    frame_idx = 0
    while frame_idx < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        results = yolo_model.track(source=frame, imgsz=640, conf=0.6, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            for box, score, class_id in zip(boxes, scores, class_ids):
                if int(class_id) != 0:  # face class
                    continue
                x1, y1, x2, y2 = map(int, box[:4])
                detections.append([[x1, y1, x2 - x1, y2 - y1], score, int(class_id)])

        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = (int(v) for v in ltrb)
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(width, xmax), min(height, ymax)

            face_crop = frame[ymin:ymax, xmin:xmax]
            if face_crop.size == 0:
                continue

            crop_path = os.path.join(tmp_dir, f"track_{track_id}.jpg")
            cv2.imwrite(crop_path, face_crop)
            aligned_face = align.get_aligned_face(crop_path)

            if aligned_face is None:
                # Off-angle / motion-blurred frame: no reliable embedding, so
                # don't print a score that would look like a real result.
                color = (160, 160, 160)  # gray
                label = f"ID {track_id}  tracking..."
            else:
                tensor = to_input(aligned_face)
                with torch.no_grad():
                    feature, _ = model(tensor)
                score = F.cosine_similarity(feature, reference_embedding).item()

                is_match = score >= args.match_threshold
                color = (0, 200, 0) if is_match else (0, 0, 220)  # BGR
                label = f"ID {track_id}  {'MATCH' if is_match else 'no match'}  {score:.2f}"

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            text_y = max(ymin - 10, 20)
            cv2.putText(frame, label, (xmin, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"  processed {frame_idx}/{args.max_frames} frames")

    cap.release()
    writer.release()
    print(f"Saved annotated demo to {args.output}")


if __name__ == '__main__':
    main()
