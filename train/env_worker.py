"""Worker process of GroupedAsyncVectorEnv.

A spawn worker imports the module of its target function. This module imports only gymnasium, so a worker does not
load torch and the CUDA libraries (about 700 MB of host memory for each worker in train/train_gym.py).
"""

import traceback

import gymnasium as gym


def grouped_env_worker(pipe, parent_pipe, env_name: str, num_envs: int):
    """Step a sync vector env of num_envs envs for GroupedAsyncVectorEnv."""

    parent_pipe.close()
    env = gym.make_vec(env_name, num_envs = num_envs, vectorization_mode = "sync")

    try:
        while True:
            command, data = pipe.recv()

            if command == "reset":
                pipe.send(("ok", env.reset(**data)))
            elif command == "step":
                pipe.send(("ok", env.step(data)))
            elif command == "close":
                break
            else:
                raise ValueError(f"unknown command {command!r}")
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception:
        pipe.send(("error", traceback.format_exc()))
    finally:
        env.close()
        pipe.close()
