import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, MaxAndSkipObservation, ResizeObservation

from rlforge.config import FRAME_SKIP, RESIZE


def pixel_pipeline(
    env: gym.Env,
    skip: int = FRAME_SKIP,
    size: tuple[int, int] = RESIZE,
    grayscale: bool = True,
) -> gym.Env:
    """Frame skip -> resize -> grayscale, for any env whose observation is an RGB image.

    Dict observations (e.g. ViZDoom) need SelectKey first.
    """
    if skip > 1:
        env = MaxAndSkipObservation(env, skip=skip)
    env = ResizeObservation(env, size)
    if grayscale:
        env = GrayscaleObservation(env, keep_dim=True)
    return env
