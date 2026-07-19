#!/usr/bin/env python3
"""Match a reference face photo against every face image in a folder
(e.g. frames extracted from a video) and show the closest match."""
import _paths  # noqa: F401
import argparse
import os

import torch
from PIL import Image

from model import load_pretrained_model, get_similarity_score, to_input
from face_alignment import align


def extract_embedding(model, image_path):
    """Extract the embedding for a given image."""
    aligned_rgb_img_input = align.get_aligned_face(image_path)
    if aligned_rgb_img_input is None:
        print(f"Error aligning image {image_path}")
        return None

    tensor = to_input(aligned_rgb_img_input)
    with torch.no_grad():
        feature, _ = model(tensor)
    return feature


def recognize_faces(reference_image_path, output_dir):
    model = load_pretrained_model('ir_101')

    reference_embedding = extract_embedding(model, reference_image_path)
    if reference_embedding is None:
        print("Error extracting embedding for the reference image.")
        return

    best_score = -1
    best_image_path = None

    for filename in os.listdir(output_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(output_dir, filename)

            face_embedding = extract_embedding(model, image_path)
            if face_embedding is None:
                continue

            similarity_score = get_similarity_score(model, image_path, reference_embedding)

            if similarity_score > best_score:
                best_score = similarity_score
                best_image_path = image_path

    if best_image_path:
        print(f"Image with the highest similarity score: {best_image_path} (Score: {best_score:.4f})")
        Image.open(best_image_path).show()
    else:
        print("No matching person found.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Find the face image in a folder that best matches a reference photo."
    )
    parser.add_argument("reference_image", help="Path to the reference face photo")
    parser.add_argument(
        "--faces-dir", default="output_faces/",
        help="Folder of candidate face images to search (default: output_faces/)",
    )
    args = parser.parse_args()
    recognize_faces(args.reference_image, args.faces_dir)
