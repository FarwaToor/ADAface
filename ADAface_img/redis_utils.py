#!/usr/bin/env python3

import numpy as np
import os
import redis

# Initialize Redis client
r = redis.Redis(host='localhost', port=6379, db=0)

# Directory containing embeddings
embedding_dir = '/home/farwa/ADAFACE_rec/embeddings'

def get_next_person_id():
    """
    Generate the next unique person ID.
    """
    current_id = r.get('current_person_id')
    if current_id is None:
        r.set('current_person_id', 1)
        return 1
    else:
        next_id = int(current_id) + 1
        r.set('current_person_id', next_id)
        return next_id

def store_all_embeddings(embedding_dir):
    """
    Store all embeddings from the specified directory into Redis.
    Clears the Redis database before storing new embeddings.
    """
    r.flushdb()  # Clear the database before storing new embeddings

    for filename in os.listdir(embedding_dir):
        if filename.endswith('.npy'):
            person_id = get_next_person_id()
            embedding_path = os.path.join(embedding_dir, filename)
            embedding = np.load(embedding_path)
            person_name = filename.replace('_embeddings.npy', '')
            r.hset(f'person_{person_id}', mapping={
                'embedding': embedding.tobytes(),
                'person_id': person_id,
                'person_name': person_name
            })
            print(f"Stored data for 'person_{person_id}' with name '{person_name}'")

def print_all_data():
    """
    Print all data stored in Redis.
    """
    for key in r.keys():
        key_type = r.type(key).decode('utf-8')
        if key_type == 'hash':
            person_id = key.decode('utf-8')
            data = r.hgetall(person_id)
            person_name = data.get(b'person_name', b'Unknown').decode('utf-8')
            print(f"Data for '{person_id}' (Name: '{person_name}'): {data}")
        else:
            print(f"Skipping key '{key.decode('utf-8')}', which is of type '{key_type}'")

def load_all_embeddings():
    """
    Load all embeddings from Redis.
    """
    all_embeddings = {}
    for key in r.keys():
        key_type = r.type(key).decode('utf-8')
        if key_type == 'hash':
            person_id = key.decode('utf-8')
            data = r.hgetall(person_id)
            embedding_bytes = data.get(b'embedding')
            if embedding_bytes:
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                all_embeddings[person_id] = embedding
        else:
            print(f"Skipping key '{key.decode('utf-8')}', which is of type '{key_type}'")
    return all_embeddings
    
def delete_embedding(person_id):
    """
    Delete embedding from Redis.
    """
    key = f'person_{person_id}'
    r.delete(key)
    print(f"Deleted data for 'person_{person_id}'")

# Example usage
if __name__ == "__main__":
    # Store embeddings
    store_all_embeddings(embedding_dir)

    # Print all data
    print_all_data()

    # Load and print all embeddings
    embeddings = load_all_embeddings()
    print(embeddings)

