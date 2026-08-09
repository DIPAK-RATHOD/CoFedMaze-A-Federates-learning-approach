"""
task_similarity.py

Computes Task Similarity (TS) from a static task feature vector,
recomputed only every K episodes.
"""

from dataclasses import dataclass
import torch

from env.generator.generator_factory import available_generators
from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv
from knowledge_graph.normalization import clip_normalize

_ALGORITHMS = available_generators()

_ROWS_COLS_RANGE = (3, 51)
_WINDOW_SIZE_RANGE = (3, 15)
_OBJECT_COUNT_RANGE = (0, 10)


@dataclass
class TaskFeatures:
    """A single node's task feature vector, extracted from its CoFedMazeParallelEnv config."""
    rows: int
    cols: int
    algorithm: str
    window_size: int
    num_checkpoints: int
    num_obstacles: int
    num_key_door_pairs: int

    @classmethod
    def from_env(cls, env: CoFedMazeParallelEnv) -> "TaskFeatures":
        return cls(
            rows=env.rows, cols=env.cols, algorithm=env.algorithm, window_size=env.window_size,
            num_checkpoints=env.num_checkpoints, num_obstacles=env.num_obstacles,
            num_key_door_pairs=env.num_key_door_pairs,
        )

    def to_vector(self) -> torch.Tensor:
        rows_n = clip_normalize(self.rows, *_ROWS_COLS_RANGE)
        cols_n = clip_normalize(self.cols, *_ROWS_COLS_RANGE)
        window_n = clip_normalize(self.window_size, *_WINDOW_SIZE_RANGE)
        checkpoints_n = clip_normalize(self.num_checkpoints, *_OBJECT_COUNT_RANGE)
        obstacles_n = clip_normalize(self.num_obstacles, *_OBJECT_COUNT_RANGE)
        key_doors_n = clip_normalize(self.num_key_door_pairs, *_OBJECT_COUNT_RANGE)
        algo_onehot = [1.0 if self.algorithm == a else 0.0 for a in _ALGORITHMS]

        return torch.tensor(
            [rows_n, cols_n, window_n, checkpoints_n, obstacles_n, key_doors_n, *algo_onehot],
            dtype=torch.float32,
        )


def compute_task_similarity(features_a: TaskFeatures, features_b: TaskFeatures) -> float:
    """
    Cosine similarity between two nodes' NORMALIZED task feature
    vectors, remapped from [-1, 1] onto [0, 1]. Returns 0.5 (neutral)
    if either vector has 0 norm.
    """
    vec_a = features_a.to_vector()
    vec_b = features_b.to_vector()

    norm_a = vec_a.norm().item()
    norm_b = vec_b.norm().item()
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.5

    cos_sim = torch.dot(vec_a, vec_b).item() / (norm_a * norm_b)
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return (cos_sim + 1.0) / 2.0


if __name__ == "__main__":
    from env.wrappers.pettingzoo_env import CoFedMazeParallelEnv

    env_a = CoFedMazeParallelEnv(rows=9, cols=9, algorithm="recursive_backtracking", num_checkpoints=2)
    env_b = CoFedMazeParallelEnv(rows=9, cols=9, algorithm="recursive_backtracking", num_checkpoints=2)

    features_a = TaskFeatures.from_env(env_a)
    features_b = TaskFeatures.from_env(env_b)

    ts_identical = compute_task_similarity(features_a, features_b)
    assert abs(ts_identical - 1.0) < 1e-6

    print("OK")
