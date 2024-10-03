import json
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def get_subject_mapping(filename: str = "subject_mapping.json") -> dict:
    path = os.path.join(BASE_PATH, filename)
    with open(path) as f:
        return json.load(f)
