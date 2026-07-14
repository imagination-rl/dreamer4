import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import train.train_gym as train_gym
from train.profiling import (
    PROFILE_MEASURE_STEPS,
    PROFILE_TOTAL_STEPS,
    PROFILE_WARMUP_STEPS,
    TrainingProfiler,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "train" / "config.yaml"


class FakeWriter:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.scalars = []
        self.flushes = 0
        self.closed = False

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        self.flushes += 1

    def close(self):
        self.closed = True


class FakeTorchProfiler:
    def __init__(self):
        self.starts = 0
        self.steps = 0
        self.stops = 0

    def start(self):
        self.starts += 1

    def step(self):
        self.steps += 1

    def stop(self):
        self.stops += 1


def test_lightweight_timings_use_shared_writer_and_distinct_tensorboard_tags(tmp_path):
    writer = FakeWriter(tmp_path)
    profiler = TrainingProfiler(
        tmp_path,
        torch.device("cpu"),
        writer = writer,
        collect_timings = True,
    )

    with profiler.record("world_model/model/forward"):
        torch.ones(2) + 1

    profiler.close()

    assert profiler.log_dir == tmp_path
    assert profiler.writer.scalars[0][0] == "timings/world_model/model/forward/cpu_milliseconds"
    assert profiler.writer.scalars[0][1] >= 0.
    assert profiler.writer.scalars[0][2] == 0
    assert not profiler.writer.closed


def test_cuda_scopes_record_cpu_and_cuda_timings(tmp_path, monkeypatch):
    events = []

    class FakeEvent:
        def __init__(self, *, enable_timing):
            assert enable_timing
            events.append(self)

        def record(self):
            pass

        def elapsed_time(self, other):
            assert other in events
            return 12.5

    monkeypatch.setattr("train.profiling.torch.cuda.Event", FakeEvent)
    monkeypatch.setattr("train.profiling.torch.cuda.synchronize", lambda _: None)

    writer = FakeWriter(tmp_path)
    profiler = TrainingProfiler(
        tmp_path,
        torch.device("cuda"),
        writer = writer,
        collect_timings = True,
    )
    with profiler.record("rollout/replay/store"):
        pass
    with profiler.record("world_model/model/forward", cuda = True):
        pass
    profiler.close()

    tags = [tag for tag, _, _ in profiler.writer.scalars]
    assert "timings/rollout/replay/store/cpu_milliseconds" in tags
    assert "timings/rollout/replay/store/cuda_milliseconds" not in tags
    assert "timings/world_model/model/forward/cpu_milliseconds" in tags
    assert "timings/world_model/model/forward/cuda_milliseconds" in tags


def test_profile_mode_is_lazy_and_exits_after_five_plus_four_steps(tmp_path, monkeypatch):
    fake_profiler = FakeTorchProfiler()
    schedule_kwargs = {}
    profile_kwargs = {}

    def fake_schedule(**kwargs):
        schedule_kwargs.update(kwargs)
        return "schedule"

    def fake_profile(**kwargs):
        profile_kwargs.update(kwargs)
        return fake_profiler

    monkeypatch.setattr("train.profiling.torch.profiler.schedule", fake_schedule)
    monkeypatch.setattr("train.profiling.torch.profiler.profile", fake_profile)
    monkeypatch.setattr("train.profiling.torch.profiler.tensorboard_trace_handler", lambda _: "trace_handler")

    profiler = TrainingProfiler(
        tmp_path,
        torch.device("cpu"),
        collect_timings = False,
        profile = True,
        profile_memory = True,
        profile_record_shapes = True,
        profile_with_stack = True,
        profile_with_flops = True,
    )
    assert fake_profiler.starts == 0

    for _ in range(PROFILE_TOTAL_STEPS - 1):
        with profiler.step("world_model/step"):
            pass

    with pytest.raises(SystemExit) as exc:
        with profiler.step("world_model/step"):
            pass

    assert exc.value.code == 0
    assert schedule_kwargs == {
        "wait": 0,
        "warmup": PROFILE_WARMUP_STEPS,
        "active": PROFILE_MEASURE_STEPS,
        "repeat": 1,
    }
    assert profile_kwargs["profile_memory"] is True
    assert profile_kwargs["record_shapes"] is True
    assert profile_kwargs["with_stack"] is True
    assert profile_kwargs["with_flops"] is True
    assert fake_profiler.starts == 1
    assert fake_profiler.steps == PROFILE_TOTAL_STEPS
    assert fake_profiler.stops == 1
    assert profiler.closed


def test_real_profiler_emits_one_tensorboard_trace_on_ninth_step(tmp_path):
    trace_calls = []
    trace_handler = torch.profiler.tensorboard_trace_handler(str(tmp_path / "captured"))

    def on_trace_ready(profiler):
        trace_calls.append(profiler.step_num)
        trace_handler(profiler)

    profiler = TrainingProfiler(tmp_path, torch.device("cpu"), collect_timings = False, profile = True)
    profiler.profiler.on_trace_ready = on_trace_ready

    for _ in range(PROFILE_TOTAL_STEPS - 1):
        with profiler.step("world_model/step"):
            torch.ones(2) + 1

    with pytest.raises(SystemExit) as exc:
        with profiler.step("world_model/step"):
            torch.ones(2) + 1

    assert exc.value.code == 0
    assert trace_calls == [PROFILE_TOTAL_STEPS]
    assert len(list((tmp_path / "captured").glob("*.pt.trace.json"))) == 1
    assert profiler.closed

    trace_path = next((tmp_path / "captured").glob("*.pt.trace.json"))
    trace = json.loads(trace_path.read_text())
    assert any(event.get("name") == "world_model/step" for event in trace["traceEvents"])


class FakeEnvironment:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeReplay:
    episode_lens = torch.tensor([1.])

    def store_episode(self, **kwargs):
        pass

    def __len__(self):
        return 1


class FakeSingleExperience:
    def cpu(self):
        return self

    def to_buffer_dict(self):
        return {"latents": torch.zeros(1, 1)}, {}


class FakeExperience:
    episode_return = torch.zeros(1)
    rewards = torch.zeros(1, 1)
    terminals = torch.zeros(1, 1, dtype = torch.bool)
    is_truncated = torch.zeros(1, 1, dtype = torch.bool)

    def cpu(self):
        return self

    def unbind(self):
        return [FakeSingleExperience()]


class FakeTokenizer(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))


class FakeWorldModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.world_weight = torch.nn.Parameter(torch.zeros(()))
        self.agent_weight = torch.nn.Parameter(torch.zeros(()))

    def world_model_parameters(self):
        return [self.world_weight]

    def agent_parameters(self):
        return [self.agent_weight]

    def muon_parameters(self):
        return []

    def interact_with_env(self, *args, **kwargs):
        return FakeExperience()


def install_fake_main_runtime(monkeypatch):
    environment = FakeEnvironment()
    replay = FakeReplay()
    profiler_instances = []

    class RecordingProfiler(TrainingProfiler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            profiler_instances.append(self)

    def fake_train_world_model(*args, steps, global_step, training_profiler, **kwargs):
        for _ in range(steps):
            with training_profiler.step("world_model/step"):
                with training_profiler.record("world_model/model/forward", cuda = True):
                    torch.ones(2) + 1
        return global_step + steps, {"world_model/loss": 0.}

    monkeypatch.setattr(train_gym, "TrainingProfiler", RecordingProfiler)
    monkeypatch.setattr(train_gym, "inspect_env_spaces", lambda *args: (2, 1, (-1., 1.)))
    monkeypatch.setattr(train_gym, "collect_random_observations", lambda *args, **kwargs: torch.zeros(1, 2))
    monkeypatch.setattr(train_gym, "train_tokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_gym, "ObservationTokenizer", FakeTokenizer)
    monkeypatch.setattr(train_gym, "DynamicsWorldModel", FakeWorldModel)
    monkeypatch.setattr(train_gym, "make_optimizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(train_gym, "make_env", lambda *args, **kwargs: environment)
    monkeypatch.setattr(train_gym, "get_episode_count", lambda env: 0)
    monkeypatch.setattr(train_gym, "get_completed_episode_stats", lambda *args: ([], []))
    monkeypatch.setattr(train_gym.Experience, "create_memmap_replay_buffer", lambda *args, **kwargs: replay)
    monkeypatch.setattr(train_gym, "train_world_model", fake_train_world_model)
    monkeypatch.setattr(
        train_gym,
        "train_agent_in_imagination",
        lambda *args, global_step, **kwargs: (global_step, {}),
    )
    monkeypatch.setattr(
        train_gym,
        "build_compile_runtime",
        lambda *args, **kwargs: SimpleNamespace(
            timings = [],
            interact_forward = None,
            world_model_loss = None,
            world_model_loss_timing = None,
            world_model_step_timing = None,
            generate_rollout = None,
            learn_from_dream = None,
            learn_from_dream_timing = None,
            imagination_step_timing = None,
        ),
    )

    return environment, profiler_instances


def main_kwargs(tmp_path, run_name, *, profile, world_model_train_steps):
    kwargs = yaml.safe_load(CONFIG_PATH.read_text())
    kwargs.update(
        log_dir = str(tmp_path),
        run_name = run_name,
        run_details = "profiling integration test",
        cpu = True,
        require_cuda = False,
        use_tensorboard = False,
        compile = False,
        compile_interact = False,
        compile_world_model = False,
        compile_generate = False,
        compile_learn = False,
        num_loops = 1,
        rollouts_per_loop = 1,
        num_envs = 1,
        max_timesteps = 1,
        pretrain_tokenizer_steps = 0,
        pretrain_tokenizer_observations = 1,
        pretrain_world_model_finetune_steps = 0,
        pretrain_world_model_combined_steps = 0,
        world_model_train_steps = world_model_train_steps,
        imagination_train_steps = 0,
        tokenizer_eval_every = 0,
        checkpoint_every = 0,
        objective = "ppo",
        profile = profile,
    )
    return kwargs


def test_main_writes_timings_profiles_nine_steps_and_closes_resources(tmp_path, monkeypatch):
    environment, profiler_instances = install_fake_main_runtime(monkeypatch)

    normal_dir = tmp_path / "normal"
    train_gym.main(**main_kwargs(tmp_path, "normal", profile = False, world_model_train_steps = 1))

    event_accumulator = EventAccumulator(str(normal_dir)).Reload()
    scalar_tags = event_accumulator.Tags()["scalars"]
    assert "timings/rollout/replay/store/cpu_milliseconds" in scalar_tags
    assert "timings/world_model/model/forward/cpu_milliseconds" in scalar_tags
    assert "timings/rollout/environment/reset/cpu_milliseconds" in scalar_tags
    assert not (normal_dir / "timings").exists()
    assert profiler_instances[-1].closed
    assert environment.closed

    environment.closed = False
    profile_dir = tmp_path / "profile" / "profile"
    with pytest.raises(SystemExit) as exc:
        train_gym.main(**main_kwargs(tmp_path, "profile", profile = True, world_model_train_steps = 9))

    assert exc.value.code == 0
    assert profiler_instances[-1].profile_step_count == PROFILE_TOTAL_STEPS
    assert profiler_instances[-1].closed
    assert environment.closed
    assert len(list(profile_dir.glob("*.pt.trace.json"))) == 1


def test_main_rejects_profile_runs_with_fewer_than_nine_model_steps(tmp_path):
    kwargs = main_kwargs(tmp_path, "too-short", profile = True, world_model_train_steps = 8)

    with pytest.raises(ValueError, match = "requires at least 9 configured"):
        train_gym.main(**kwargs)
