from typing import Callable, Optional

import gym
import numpy as np

from epic import types


class GymSampler(types.Sampler):
    def __init__(self, space: gym.Space, n_samples: int):
        self.space = space
        self.n_samples = n_samples

    def sample(self):
        return np.array([self.space.sample() for _ in range(self.n_samples)])


class PreloadedDataSampler(types.Sampler):
    """A sampler that samples from a preloaded dataset."""
    data: np.ndarray
    n_samples: Optional[int]
    rng: np.random.Generator
    def __init__(self, data: np.ndarray, n_samples: Optional[int] = None, rng: Optional[np.random.Generator] = None):
        """Initializes the sampler.

        Args:
            data: The data to sample from.
            n_samples: The number of samples to take from the data. If None, all data is used
                (the array is returned with no modifications).
                If a value is specified, the first axis of the array is the dimension that the data
                is sampled from.

        Raises:
            ValueError: If the n_samples is larger than the first dimension of the data.
            """
        self.data = data
        self.n_samples = n_samples
        self.rng = rng or np.random.default_rng()

        if self.n_samples is not None and self.data.shape[0] < self.n_samples:
            raise ValueError(f"n_samples ({self.n_samples}) must be less than the number of data points ({self.data.shape[0]})")

    def sample(self):
        return self.data[self.rng.integers(0, self.data.shape[0], self.n_samples)] \
                if self.n_samples is not None \
                else self.data


def make_sampler(fn: Callable[[], np.ndarray]) -> types.Sampler:
    """Decorator to create a sampler from a function.

    The function should take no arguments and return a numpy array.

    Args:
        fn: The function to use to create the sampler.

    Returns: A sampler instance.
    """
    class FunctionSampler(types.Sampler):
        def sample(self) -> np.ndarray:
            return fn()

    return FunctionSampler()
