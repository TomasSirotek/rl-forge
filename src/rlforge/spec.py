"""The plug socket: everything `core/` needs to know about a game.

Adding a game means writing one of these in `games/<name>/game.py`.
Nothing in `core/` imports a game -- the spec is always passed in.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class GameSpec:
    # Short slug. Picks the log/tensorboard folders: logs/<name>, board/<name>.
    name: str

    # Gym id the game is registered under, e.g. 'SuperMarioBros-1-1-v0'.
    env_id: str

    # env -> wrapped env. Owns EVERYTHING game-specific: the action-space
    # remap (JoypadSpace is nes-py only), resize, grayscale, frame skip.
    # Training and play both call this, so whatever training saw, play sees.
    preprocess: Callable[[Any], Any]

    # info[] key that StallLimit watches to decide the agent is stuck.
    progress_key: str = "x_pos"

    # Extra info[] keys printed after each episode during play.
    report_keys: tuple = ("x_pos", "flag_get")

    # How many frames the policy stacks to perceive motion. Must match
    # between training and play or the loaded model sees the wrong shape.
    frame_stack: int = 4

    # Extra gym.make() options used only by play(), e.g. ViZDoom's
    # {"screen_resolution": "RES_1280X720", "sound_enabled": True}.
    # Training ignores them: frames get resized to RESIZE either way, and
    # windows/sound would only slow the parallel envs down.
    play_kwargs: Mapping[str, Any] = field(default_factory=dict)
