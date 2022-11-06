import os

import yaml


def _load_cfg(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


EMBEDDING_CFG = _load_cfg(os.path.join("embedding_cfg.yaml"))
