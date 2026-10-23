import numpy as np
import torch

from memmap_replay_buffer.replay_buffer import ReplayDatasetTimeWindow

from dreamer4.dreamer4 import Experience
from train import offline_pretraining as offline
from train import train_gym

ACTION_RANGE = (-1., 1.)
LENGTH = 60


def episode(flip_rows, seed):
    """One episode of LENGTH transitions. The root angle is pi (flipped) on the rows of each [start, stop) in flip_rows."""

    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((LENGTH + 1, 17)).astype(np.float32)
    obs[:, train_gym.FLIP_ONSET_ANGLE_INDEX] = 0.1
    for start, stop in flip_rows:
        obs[start:stop, train_gym.FLIP_ONSET_ANGLE_INDEX] = np.pi
    return offline.offline_episode_experience(
        latents = torch.zeros(LENGTH + 1, 2, 8),
        observations = obs,
        env_actions = rng.uniform(-0.9, 0.9, size = (LENGTH, 6)).astype(np.float32),
        rewards = rng.standard_normal(LENGTH).astype(np.float32),
        terminated = False,
        truncated = True,
        action_range = ACTION_RANGE,
        step_size = 16,
    )


def test_flip_onset_windows_hold_the_onset(tmp_path):
    first = [episode([(30, LENGTH + 1)], 0), episode([], 1), episode([(10, 15), (50, 55)], 2)]
    replay = Experience.create_memmap_replay_buffer(first[0], tmp_path / "replay", max_episodes = 8, max_timesteps = LENGTH + 1, circular = True)
    for experience in first:
        offline.store_experience(replay, experience)

    # Onset rows: the first flipped row of each upright-to-flipped transition. A recovery is not an onset.
    onsets = train_gym.flip_onsets_of(replay)
    assert sorted(zip(onsets.slots.tolist(), onsets.rows.tolist())) == [(0, 30), (2, 10), (2, 50)]

    # The next call reads only the new slot, and the replay keeps the same index object.
    offline.store_experience(replay, episode([(5, LENGTH + 1)], 3))
    assert train_gym.flip_onsets_of(replay) is onsets
    assert sorted(zip(onsets.slots.tolist(), onsets.rows.tolist())) == [(0, 30), (2, 10), (2, 50), (3, 5)]

    # Fraction 0 is the normal dataset.
    assert type(train_gym.replay_time_windows(replay, 20)) is ReplayDatasetTimeWindow

    torch.manual_seed(0)
    angle = train_gym.FLIP_ONSET_ANGLE_INDEX
    windows = train_gym.replay_time_windows(replay, 20, flip_onset_fraction = 0.5)
    # Only flip-onset windows for the checks below (the flag accepts fractions below 1).
    windows.fraction = 1.

    # A flip-onset item and a normal item collate into one batch: same keys, shapes, and dtypes.
    normal = ReplayDatasetTimeWindow.__getitem__(windows, 0)
    onset = windows.window(0, 15)
    assert {key: (value.shape, value.dtype) for key, value in normal.items()} == {key: (value.shape, value.dtype) for key, value in onset.items()}
    assert torch.equal(onset["critic_state"], torch.from_numpy(np.asarray(replay.data["critic_state"][0, 15:35])))
    for index in range(200):
        item = windows[index % len(windows)]
        flipped = np.abs(item["critic_state"][:, angle].numpy()) > np.pi / 2
        # World-model windows: both rows of the transition are in the window.
        assert int(item["_lens"]) == 20 and (~flipped[:-1] & flipped[1:]).any()

    # Prompt windows: 8 upright rows that end 1 to 32 rows before the onset (the onset falls in the dream steps).
    prompts = train_gym.replay_time_windows(replay, 8, flip_onset_fraction = 0.5, min_before = 8, max_before = 8 + 32 - 1, return_indices = True)
    prompts.fraction = 1.
    for index in range(200):
        item = prompts[index % len(prompts)]
        end = int(item["_start"]) + 8
        rows = {(0, 30), (2, 10), (2, 50), (3, 5)}
        candidates = [row for slot, row in rows if slot == int(item["_index"]) and end <= row <= end + 31]
        # The onset at row 5 is in the first 8 rows: its prompt starts at 0 and holds the onset.
        assert candidates or (int(item["_index"]), int(item["_start"])) == (3, 0)
