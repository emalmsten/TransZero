# src/trans_zero/utils/paths.py

import pathlib

# Find project root by walking up from this file
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"