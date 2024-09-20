#!/usr/bin/env python3

from PIL import Image
from face_alignment import align

def load_image(image_path):
    try:
        img = Image.open(image_path)
        img.verify()  # Verify that the image file is intact
        img = Image.open(image_path)  # Reopen the image for processing
        print(f"Loaded image from {image_path}")
        return img
    except (IOError, SyntaxError) as e:
        print(f"Error opening image file {image_path}: {e}")
        return None


