"""
Implements Divergence-Free Rewards Distance Calculation.
"""

from typing import Optional, TypeVar

import numpy as np
import numpy.typing as npt

from epic import samplers, types, utils
from epic.distances import base, pearson_mixin
import torch
import torch.nn as nn
import torch.optim as optim

T_co = TypeVar("T_co", covariant=True)


class DivergenceFree(base.Distance, pearson_mixin.PearsonMixin):
    default_samples_cov = 500
    default_samples_can = 500

    def __init__(
        self,
        discount_factor: float,
        state_sampler: Optional[samplers.BaseSampler[samplers.StateSample]] = None,
        action_sampler: Optional[samplers.BaseSampler[npt.NDArray]] = None,
        coverage_sampler: Optional[samplers.BaseSampler[samplers.CoverageSample]] = None,
    ):
        """Initialize the Divergence-Free Reward Distance.

        Args:
          state_sampler: The sampler for the state distribution. Optional if the coverage_sampler is provided.
          action_sampler: The sampler for the action distribution. Optional if the coverage_sampler is provided.
          coverage_sampler: The sampler for the coverage distribution. If not given,
            a default sampler is constructed as drawing from the product
            distribution induced by the distributions of state and action.
          discount_factor: The discount factor.
        """
        if coverage_sampler is None:
          assert state_sampler is not None and action_sampler is not None "If no coverage sampler is given, state and action samplers must be provided."
        else:
          assert state_sampler is None and action_sampler is None "If a coverage sampler is given, state and action samplers will not be used."
        coverage_sampler = coverage_sampler or samplers.ProductDistrCoverageSampler(action_sampler, state_sampler)
        super().__init__(discount_factor, state_sampler, action_sampler, coverage_sampler)
        state_sample, action_sample, _, _ = self.coverage_sampler.sample(1)
        self.state_dim = state_sample.shape[1]
        self.action_dim = action_sample.shape[1]

    def canonicalize(
      self,
      reward_function: types.RewardFunction,
      /,
      n_samples_can: Optional[int],
    ) -> types.RewardFunction:
        """Canonicalizes a reward function into a divergence-free reward
        function of the same equivalence class.

        This is done by fitting a potential function (constructed from a neural
        network) to minimize the L2 norm of the shaped reward function.
        """
        rew_fn = utils.multidim_rew_fn(reward_function)
        n_samples_can = n_samples_can or self.default_samples_can
        assert isinstance(n_samples_can, int)

        self.net = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim * 4),
            nn.ReLU(),
            nn.Linear(self.state_dim * 4, 1),
        )

        optimizer = optim.AdamW(self.net.parameters())

        state_sample, action_sample, next_state_sample, done_sample = self.coverage_sampler.sample(n_samples_can)
        state_sample = torch.from_numpy(state_sample).float()
        action_sample = torch.from_numpy(action_sample).float()
        next_state_sample = torch.from_numpy(next_state_sample).float()
        done_sample = torch.from_numpy(done_sample).float()
        n_epochs = 50000

        for _ in range(n_epochs):
            optimizer.zero_grad()
            potential = self.discount_factor * self.net(next_state_sample) - self.net(state_sample)
            l2_loss = torch.mean((torch.from_numpy(rew_fn(state_sample, action_sample, next_state_sample, done_sample)).float() + potential) ** 2)
            l2_loss.backward()
            optimizer.step()

        def canonical_reward_fn(state, action, next_state, done, /):
            """Divergence-Free canonical reward function.

            Args:
                state: The batch of state samples from the coverage distribution.
                action: The batch of action samples from the coverage distribution.
                next_state: The batch of next state samples from the coverage distribution.
                done: The batch of done samples from the coverage distribution.

            Returns:
                The canonicalized reward function.
            """
            n_samples_cov = state.shape[0]
            assert n_samples_cov == action.shape[0] == next_state.shape[0] == done.shape[0]
            return rew_fn(state, action, next_state, done) + self.discount_factor * self.net(torch.from_numpy(next_state).float()) - self.net(torch.from_numpy(state).float())

        return canonical_reward_fn

def divergence_free_distance(x, y, /, *, state_sampler, action_sampler, coverage_sampler, discount_factor, n_samples_cov: int, n_samples_can: int):
  """Calculates the divergence-free reward distance between two reward functions.

  Helper function that automatically instantiates the DivergenceFree class and computes the distance
  between two reward functions using its canonicalization.

  Args:
    x: The first reward function.
    y: The second reward function.
    state_sampler: The sampler for the state distribution. Optional if the coverage_sampler is provided.
    action_sampler: The sampler for the action distribution. Optional if the coverage_sampler is provided.
    coverage_sampler: The sampler for the coverage distribution. If not given,
      a default sampler is constructed as drawing from the product
      distribution induced by the distributions of state and action.
    discount_factor: The discount factor.
    n_samples_cov: The number of samples to use for the coverage distance.
    n_samples_can: The number of samples to use for the canonicalization.
    """
  return DivergenceFree(discount_factor, state_sampler, action_sampler, coverage_sampler).distance(x, y, n_samples_cov, n_samples_can)

