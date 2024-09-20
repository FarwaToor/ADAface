#!/usr/bin/env python3
import numpy as np
import torch
from AdaFace import net
from config import ADA_MODEL_PATH
from face_alignment import align

# Define the adaface_models dictionary
adaface_models = {
    'ir_101': "AdaFace/pretrained/adaface_ir101_webface12m.ckpt",
}

def load_pretrained_model(architecture='ir_101'):
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    print(f"Loading model from {ADA_MODEL_PATH}")
    statedict = torch.load(ADA_MODEL_PATH, weights_only=True)['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor(np.array([brg_img.transpose(2, 0, 1)])).float()
    return tensor

def get_similarity_score(model, image_path, embedding_tensor):
    aligned_rgb_img_input = align.get_aligned_face(image_path)
    if aligned_rgb_img_input is None:
        print("Error: Aligned image is None")
        return 0
    bgr_tensor_input = to_input(aligned_rgb_img_input)
    feature_input, _ = model(bgr_tensor_input)
    similarity_score = torch.nn.functional.cosine_similarity(feature_input, embedding_tensor)
    return similarity_score.item()


