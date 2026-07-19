#!/usr/bin/env python3
"""Flask web app: enroll face photos, search for a match, browse/delete records."""
import _paths  # noqa: F401  (adds common/ to sys.path — must run before the imports below)

import os

import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from config import UPLOAD_DIR, EMBEDDING_DIR
from face_alignment import align
from image_utils import load_image
from model import load_pretrained_model, to_input, get_similarity_score
from redis_utils import store_all_embeddings, load_all_embeddings, get_next_person_id, delete_embedding
from sqlite_utils import (
    create_table_if_not_exists,
    store_person,
    retrieve_all_persons,
    delete_person as delete_person_from_sqlite,
)

app = Flask(__name__)

# Initialize the SQLite3 database
create_table_if_not_exists()

# Load the model
model = load_pretrained_model('ir_101')


def find_best_match(input_image_path):
    input_image = load_image(input_image_path)
    if input_image is None:
        return None, 0

    all_embeddings = load_all_embeddings()
    if not all_embeddings:
        return None, 0

    aligned_rgb_img_input = align.get_aligned_face(input_image_path)
    if aligned_rgb_img_input is None:
        return None, 0

    bgr_tensor_input = to_input(aligned_rgb_img_input)
    feature_input, _ = model(bgr_tensor_input)

    best_score = -1
    best_id = None
    for person_id, embedding in all_embeddings.items():
        embedding_tensor = torch.tensor([embedding]).float()
        similarity_score = get_similarity_score(model, input_image_path, embedding_tensor)

        if similarity_score > best_score:
            best_score = similarity_score
            best_id = person_id

    return best_id, best_score


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/enroll', methods=['GET', 'POST'])
def enroll():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_DIR, filename)
        file.save(image_path)

        input_image = load_image(image_path)
        if input_image is None:
            return jsonify({"error": "Error loading image"}), 400

        aligned_rgb_img_input = align.get_aligned_face(image_path)
        if aligned_rgb_img_input is None:
            return jsonify({"error": "Error aligning image"}), 400

        bgr_tensor_input = to_input(aligned_rgb_img_input)
        feature_input, _ = model(bgr_tensor_input)
        embedding = feature_input.detach().numpy().flatten()

        person_id = get_next_person_id()
        person_name = os.path.splitext(filename)[0]  # file name (no extension) is used as the display name

        embedding_path = os.path.join(EMBEDDING_DIR, f'person_{person_id}_embeddings.npy')
        np.save(embedding_path, embedding)

        store_all_embeddings(str(EMBEDDING_DIR))
        store_person(person_id, person_name)

        return jsonify({"message": f"Image enrolled successfully with person_id {person_id}"}), 200

    return render_template('enroll.html')


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_DIR, filename)
        file.save(image_path)

        best_id, best_score = find_best_match(image_path)
        if best_id is None:
            return jsonify({"message": "No matching person found"}), 404

        return jsonify({"person_id": best_id, "score": best_score}), 200

    return render_template('search.html')


@app.route('/show_all_persons')
def show_all_persons():
    persons = retrieve_all_persons()
    return render_template('persons.html', persons=persons)


@app.route('/delete_person', methods=['GET', 'POST'])
def delete_person():
    if request.method == 'POST':
        person_id = request.form.get('person_id')
        if not person_id:
            return render_template('delete.html', message="No person ID provided")

        person_id = int(person_id)
        try:
            delete_embedding(person_id)
            delete_person_from_sqlite(person_id)
            message = f"Person with ID {person_id} deleted successfully"
        except Exception as e:
            message = f"Error deleting person with ID {person_id}: {str(e)}"

        return render_template('delete.html', message=message)

    return render_template('delete.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4996)
