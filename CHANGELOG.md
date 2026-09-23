# Changelog

All notable changes to rlforge. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `pixel_pipeline(env, skip, size, grayscale)`: frame skip → resize → grayscale in one call, for any env with RGB image observations.
- `SelectKey(env, key)`: wrapper that keeps one entry of a dictionary observation (e.g. ViZDoom's `"screen"`), so it can go through `pixel_pipeline`.
- `GameSpec.play_kwargs`: extra `gym.make` options used only by `play()` (window size, sound, ...).

### Removed
- `GameSpec.screen_resolution`: pass it in `play_kwargs` instead, e.g. `play_kwargs={"screen_resolution": "RES_1280X720"}`.

## [0.1.0]

### Added
- `GameSpec`: describes a game (Gymnasium env id, preprocessing, progress and report keys, frame stack, play window size).
- `train(spec, cfg)`: PPO training with parallel envs, evaluation, `best_model.zip` / `final_model.zip` checkpoints and automatic resume.
- `play(spec, model_path, fps)`: watch a trained agent; falls back to the newest final model, then to random actions.
- `TrainConfig`: training hyperparameters.
- `StallLimit`: wrapper that ends an episode when the agent stops making progress.
- `RenderModes`: `WINDOW_RENDER` and `RGB_ARRAY`.
