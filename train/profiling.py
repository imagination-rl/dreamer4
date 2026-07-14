"""Lightweight timings and bounded torch.profiler support for training."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from time import perf_counter

import torch
from torch.profiler import ProfilerActivity, record_function
from torch.utils.tensorboard import SummaryWriter


PROFILE_WARMUP_STEPS = 5
PROFILE_MEASURE_STEPS = 4
PROFILE_TOTAL_STEPS = PROFILE_WARMUP_STEPS + PROFILE_MEASURE_STEPS


class TrainingProfiler:
    """Record hierarchical timings and optionally capture a full PyTorch trace."""

    def __init__(
        self,
        log_dir: str | Path,
        device: torch.device,
        *,
        writer: SummaryWriter | None = None,
        collect_timings: bool = True,
        profile: bool = False,
        profile_memory: bool = False,
        profile_record_shapes: bool = False,
        profile_with_stack: bool = False,
        profile_with_flops: bool = False,
    ):
        self.device = torch.device(device)
        self.collect_timings = collect_timings
        self.profile_mode = profile
        self.enabled = collect_timings or profile
        self.profile_step_count = 0
        self.closed = False
        self._tag_steps = defaultdict(int)
        self._pending_cuda_timings = []

        self.log_dir = Path(log_dir)
        self.trace_dir = self.log_dir / "profile"
        self.writer = writer if collect_timings else None
        self.profiler = None
        self._profiler_started = False

        if profile:
            activities = [ProfilerActivity.CPU]
            if self.device.type == "cuda":
                activities.append(ProfilerActivity.CUDA)

            self.trace_dir.mkdir(parents = True, exist_ok = True)
            self.profiler = torch.profiler.profile(
                activities = activities,
                schedule = torch.profiler.schedule(
                    wait = 0,
                    warmup = PROFILE_WARMUP_STEPS,
                    active = PROFILE_MEASURE_STEPS,
                    repeat = 1,
                ),
                on_trace_ready = torch.profiler.tensorboard_trace_handler(str(self.trace_dir)),
                record_shapes = profile_record_shapes,
                profile_memory = profile_memory,
                with_stack = profile_with_stack,
                with_flops = profile_with_flops,
            )

    def start_step(self):
        """Start a lazily-created full profiler immediately before model work."""

        if self.profiler is not None and not self._profiler_started:
            self.profiler.start()
            self._profiler_started = True

    @contextmanager
    def record(self, name: str, *, cuda: bool = False):
        """Record CPU wall time and, for GPU work, CUDA elapsed time."""

        if not self.enabled:
            with nullcontext():
                yield
            return

        cuda_timing = self.collect_timings and cuda and self.device.type == "cuda"
        cpu_start = perf_counter() if self.collect_timings else None
        start_event = end_event = None

        if cuda_timing:
            start_event = torch.cuda.Event(enable_timing = True)
            end_event = torch.cuda.Event(enable_timing = True)
            start_event.record()

        with record_function(name):
            try:
                yield
            finally:
                if cuda_timing:
                    end_event.record()
                    self._pending_cuda_timings.append((name, start_event, end_event))
                if self.collect_timings:
                    self._write_timing(name, "cpu_milliseconds", (perf_counter() - cpu_start) * 1000.)

    @contextmanager
    def step(self, name: str, *, cuda: bool = False):
        """Record one model-training step and advance the full profiler schedule."""

        self.start_step()
        with self.record(name, cuda = cuda):
            yield

        self.advance()

    def advance(self):
        """Advance the full-profiler schedule after one complete model step."""

        if not self.profile_mode:
            return

        self.profiler.step()
        self.profile_step_count += 1

        if self.profile_step_count >= PROFILE_TOTAL_STEPS:
            self.close()
            print(
                "profile complete: "
                f"{PROFILE_WARMUP_STEPS} warmup + {PROFILE_MEASURE_STEPS} measured steps; "
                f"trace written to {self.trace_dir}"
            )
            raise SystemExit(0)

    def _write_timing(self, name: str, metric: str, milliseconds: float):
        if self.writer is None:
            return

        tag = f"timings/{name}/{metric}"
        step = self._tag_steps[tag]
        self.writer.add_scalar(tag, milliseconds, step)
        self._tag_steps[tag] += 1

    def log_timing(self, name: str, milliseconds: float, *, metric = "cpu_milliseconds"):
        """Log an already-aggregated timing using the profiler tag namespace."""

        if self.collect_timings:
            self._write_timing(name, metric, milliseconds)

    def flush_timings(self, *, flush_writer = False):
        """Resolve pending CUDA events, optionally forcing the shared writer to disk."""

        if len(self._pending_cuda_timings) > 0:
            torch.cuda.synchronize(self.device)
            pending, self._pending_cuda_timings = self._pending_cuda_timings, []
            for name, start_event, end_event in pending:
                self._write_timing(name, "cuda_milliseconds", start_event.elapsed_time(end_event))

        if flush_writer and self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.closed:
            return

        try:
            self.flush_timings(flush_writer = True)
            if self.profiler is not None and self._profiler_started:
                self.profiler.stop()
        finally:
            self.closed = True
