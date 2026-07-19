#!/usr/bin/env python3
"""Command-line face search — reuses the same enrolled data as the Flask app
(app.py) to match a photo against everyone stored in Redis, without a server."""
import _paths  # noqa: F401
import argparse

from app import find_best_match


def main():
    parser = argparse.ArgumentParser(description="Find the closest enrolled match for a face photo.")
    parser.add_argument("image_path", help="Path to the photo to search for")
    args = parser.parse_args()

    best_person, best_score = find_best_match(args.image_path)
    if best_person:
        print(f"Person with highest similarity: {best_person} (Score: {best_score})")
    else:
        print("No matching person found.")


if __name__ == '__main__':
    main()
