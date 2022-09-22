import gym
import numpy as np

from epic import types


class GymSampler(types.Sampler):
    def __init__(self, space: gym.Space, n_samples: int):
        self.space = space
        self.n_samples = n_samples

    def sample(self):
        return np.array([self.space.sample() for _ in range(self.n_samples)])
