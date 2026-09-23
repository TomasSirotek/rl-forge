# rlforge

Train a reinforcement learning agent on any [Gymnasium](https://gymnasium.farama.org/) game by writing one small spec.

rlforge handles the repetitive parts of training an agent from pixels: parallel environments, frame stacking, checkpoints, resuming, evaluation and playback. You describe the game once in a `GameSpec`, then call `train()` and `play()`. Training uses PPO from [stable-baselines3](https://stable-baselines3.readthedocs.io/).

```python
from rlforge import GameSpec, train, play

train(spec)   # learn
play(spec)    # watch it play
```

## Install

Requires Python 3.13+.

```bash
uv add "rlforge @ git+https://github.com/<you>/rlforge"
```

Add the `tensorboard` extra for training curves: `rlforge[tensorboard]`.

## Quickstart

A game is a `GameSpec`: which Gymnasium env to create, and how to preprocess it. Here is Super Mario Bros:

```python
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
import gym_super_mario_bros  # registers the Mario envs

from rlforge import GameSpec, StallLimit, pixel_pipeline, train, play


def preprocess(env):
    env = JoypadSpace(env, RIGHT_ONLY)           # 256 button combos -> 5 actions
    env = pixel_pipeline(env)                    # skip 4 frames, resize to 84x84, grayscale
    return StallLimit(env, patience=80)          # end the episode when Mario stops moving right


mario = GameSpec(
    name="mario",
    env_id="SuperMarioBros-1-1-v0",
    preprocess=preprocess,
    progress_key="x_pos",
    report_keys=("x_pos", "flag_get"),
)

train(mario)
play(mario)
```

rlforge never imports a specific game. Everything game-specific lives in your spec, so adding a new game doesn't mean changing the library.

## Preprocessing

Your `preprocess` function holds what's specific to your game. rlforge provides the parts every game repeats:

- **`pixel_pipeline(env, skip=4, size=(84, 84), grayscale=True)`**: frame skip → resize → grayscale, for any env whose observation is an RGB image. Pass `grayscale=False` if color matters for your game, or `skip=1` if the env already skips frames itself.
- **`SelectKey(env, key)`**: keeps one entry of a dictionary observation. Use it before `pixel_pipeline` for envs like ViZDoom, which return `{"screen": ..., "gamevariables": ...}`:

  ```python
  def preprocess(env):
      return pixel_pipeline(SelectKey(env, "screen"))
  ```
- **`StallLimit(env, patience, progress_key)`**: ends the episode when `info[progress_key]` stops improving.

Order matters:
1. **Game-specific changes come first**: controls remapping (like Mario's `JoypadSpace`) or `SelectKey`. They change what the env is.
2. **Then `pixel_pipeline`.**
3. **`StallLimit` goes after the frame skip**, so `patience` counts agent decisions, not raw frames.

## GameSpec

| Field | Default | What it does |
|---|---|---|
| `name` | required | Folder name for this game's logs and models |
| `env_id` | required | The Gymnasium id to create |
| `preprocess` | required | `env -> wrapped env`. Action remapping, resizing, grayscale, frame skip |
| `progress_key` | `"x_pos"` | `info[]` key that `StallLimit` watches for progress |
| `report_keys` | `("x_pos", "flag_get")` | `info[]` keys printed after each episode in `play()` |
| `frame_stack` | `4` | Frames stacked so the agent can see motion |
| `play_kwargs` | `{}` | Extra `gym.make` options used only in `play()`, e.g. `{"screen_resolution": "RES_1280X720", "sound_enabled": True}` for envs that accept them. Training ignores them |

`train()` and `play()` build the environment the same way from the spec, so the agent sees exactly the same input when playing as it did in training.

## Training

```python
from rlforge import TrainConfig, train

train(mario, TrainConfig(num_cpu=4, total_timesteps=500_000, device="cpu"))
```

`TrainConfig` holds the hyperparameters: `num_cpu`, `total_timesteps`, `learning_rate`, `n_steps`, `ent_coef`, `target_kl`, `eval_freq`, `n_eval_episodes`, `device`, `resume`.

Output goes to folders under the current directory:

```
logs/<name>/best_model.zip             saved whenever evaluation improves
logs/<name>/run_<timestamp>/final_model.zip
board/<name>/                          TensorBoard logs
```

Training **resumes automatically** from `best_model.zip` if it exists. Pass `TrainConfig(resume=False)` to start fresh.

## Playing

```python
from rlforge import play

play(mario)                                  # best model, else newest final model
play(mario, model_path="path/to/model.zip")  # a specific checkpoint
play(mario, fps=30)                          # faster playback
```

If there's no trained model yet, `play()` uses random actions, so you can check a new spec before training it.

## What's included

| Name | What it is |
|---|---|
| `GameSpec` | Describes a game: env id and preprocessing |
| `train(spec, cfg)` | Trains PPO with parallel envs, evaluation and checkpoints |
| `play(spec, model_path, fps)` | Watches a trained or random agent in a window |
| `TrainConfig` | Training hyperparameters |
| `pixel_pipeline(env, skip, size, grayscale)` | Frame skip → resize → grayscale in one call |
| `SelectKey(env, key)` | Wrapper that keeps one entry of a dictionary observation |
| `StallLimit` | Wrapper that ends an episode when the agent stops making progress |
| `RenderModes` | `WINDOW_RENDER` (`"human"`) and `RGB_ARRAY` (`"rgb_array"`) |

## License

MIT
