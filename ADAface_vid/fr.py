#!/usr/bin/env python3
import os
import torch
from PIL import Image
from model import load_pretrained_model, get_similarity_score, to_input
from face_alignment import align

# Configuration
ADA_MODEL_PATH = 'AdaFace/pretrained/adaface_ir101_webface12m.ckpt'
OUTPUT_DIR = 'output_faces/'
REFERENCE_IMAGE_PATH = '/home/farwa/ADAFACE_rec/ADAface_vid/testimg.jpg'  # Path to the reference image

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

    # Extract embedding for the reference image
    reference_embedding = extract_embedding(model, reference_image_path)
    if reference_embedding is None:
        print("Error extracting embedding for the reference image.")
        return

    best_score = -1
    best_image_path = None

    for filename in os.listdir(output_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(output_dir, filename)

            # Extract embedding for each image in the output_faces directory
            face_embedding = extract_embedding(model, image_path)
            if face_embedding is None:
                continue

            # Compute similarity score between the reference image and each face image
            similarity_score = get_similarity_score(model, image_path, reference_embedding)

            if similarity_score > best_score:
                best_score = similarity_score
                best_image_path = image_path

    if best_image_path:
        print(f"Image with the highest similarity score: {best_image_path} (Score: {best_score:.4f})")
        best_image = Image.open(best_image_path)
        best_image.show()  # Display the image with the highest score
    else:
        print("No matching person found.")

if __name__ == '__main__':
    recognize_faces(REFERENCE_IMAGE_PATH, OUTPUT_DIR)
