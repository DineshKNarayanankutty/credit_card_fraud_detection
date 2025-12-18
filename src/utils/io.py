"""
File I/O operations.
"""

import pickle
import json
from pathlib import Path
from typing import Any


def save_pickle(obj: Any, filepath: str):
    """Save object as pickle"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """Load pickle file"""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, filepath: str):
    """Save object as JSON"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(filepath: str) -> Any:
    """Load JSON file"""
    with open(filepath, "r") as f:
        return json.load(f)
