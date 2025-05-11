import json
from ..configs import FINE_JSON

def load():
    try:
        with open(FINE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save(d: dict):
    with open(FINE_JSON, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
