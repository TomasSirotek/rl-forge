import gymnasium as gym
from gymnasium import spaces

class StallLimit(gym.Wrapper):
    """End the episode if `progress_key` hasn't improved in `patience` steps."""
    def __init__(self, env, patience=80, progress_key="x_pos"):
        super().__init__(env)
        self.patience = patience
        self.progress_key = progress_key

    def reset(self, **kwargs):
        self.best, self.stale = 0, 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get(self.progress_key, 0) > self.best:
            self.best, self.stale = info[self.progress_key], 0
        else:
            self.stale += 1
            if self.stale >= self.patience:
                truncated = True
        return obs, reward, terminated, truncated, info


class SelectKey(gym.ObservationWrapper):
    """Keep one entry of a Dict observation, e.g. ViZDoom's "screen"."""

    def __init__(self, env: gym.Env, key: str):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError(f"SelectKey needs a Dict observation space, got {env.observation_space}")
        self.key = key
        self.observation_space = env.observation_space[key]

    def observation(self, observation):
        return observation[self.key]