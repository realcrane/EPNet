import os
from pathlib import Path

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BODY_DIR = os.path.join(ROOT_DIR, "body_models")
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "results", "checkpoints")
DATA_DIR = os.path.join(ROOT_DIR, "data")
TXT_DIR = os.path.join(ROOT_DIR, "data", "txt")
LOGS_DIR = os.path.join(ROOT_DIR, "results", "logs")
RENDER_DIR = os.path.join(ROOT_DIR, "results", "render")
ELASTICITY_DIR = os.path.join(ROOT_DIR, "results", "elasticity")
SUB_GRAPH_DIR = os.path.join(ROOT_DIR, "pre_processing", "subgraph")


def body_model_name(config) -> str:
    model = getattr(config.body, "model", "")
    return str(model) if model not in ("", ".", None) else "body"


def body_asset_dir(config) -> Path:
    model = getattr(config.body, "model", "")
    if model not in ("", ".", None):
        nested = Path(BODY_DIR) / str(model)
        if nested.is_dir():
            return nested
    return Path(BODY_DIR)


# Skeletons
NUM_JOINTS = {"smpl": 24, "mixamo": 65}

# Physics
GRAVITY = 9.81
