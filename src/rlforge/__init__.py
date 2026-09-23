"""Game-agnostic RL engine. 

"""

from rlforge.config import TrainConfig
from rlforge.preprocessing import pixel_pipeline
from rlforge.render_modes import RenderModes
from rlforge.rollout import play
from rlforge.spec import GameSpec
from rlforge.trainer import train
from rlforge.wrappers import SelectKey, StallLimit

__all__ = [
    "GameSpec",
    "RenderModes",
    "SelectKey",
    "StallLimit",
    "TrainConfig",
    "pixel_pipeline",
    "play",
    "train",
]
