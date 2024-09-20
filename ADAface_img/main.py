#!/usr/bin/env python3

from model import load_pretrained_model, get_similarity_score
from redis_utils import load_all_embeddings
from image_utils import load_image
from config import EMBEDDING_DIR
from face_alignment import align
from model import to_input
import torch

def find_best_match(input_image_path):
    model = load_pretrained_model('ir_101')
    input_image = load_image(input_image_path)
    if input_image is None:
        print(f"Error loading input image {input_image_path}")
        return None, 0
    all_embeddings = load_all_embeddings()
    
    if not all_embeddings:
        print("No embeddings found in Redis")
        return None, 0
    
    best_score = -1
    best_name = None

    aligned_rgb_img_input = align.get_aligned_face(input_image_path)
    if aligned_rgb_img_input is None:
        print("Error: Aligned image is None")
        return None, 0
    bgr_tensor_input = to_input(aligned_rgb_img_input)
    feature_input, _ = model(bgr_tensor_input)
    
    for person_name, embedding in all_embeddings.items():
        embedding_tensor = torch.tensor([embedding]).float()
        similarity_score = get_similarity_score(model, input_image_path, embedding_tensor)
        
        if similarity_score > best_score:
            best_score = similarity_score
            best_name = person_name
    
    return best_name, best_score

if __name__ == '__main__':
    input_image_path = '/home/farwa/ADAFACE_rec/test3.jpeg'
    best_person, best_score = find_best_match(input_image_path)
    if best_person:
        print(f"Person with highest similarity: {best_person} (Score: {best_score})")
    else:
        print("No matching person found.")

