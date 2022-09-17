from typing import Optional

import gym
import numpy as np

from epic import types, utils


class GymSampler(types.Sampler):
    def __init__(self, space: gym.Space, n_samples: int):
        self.space = space
        self.n_samples = n_samples

    def sample(self):
        return np.array([self.space.sample() for _ in range(self.n_samples)])


class EPIC:
    def __init__(
        self,
        state_sampler: types.Sampler,
        action_sampler: types.Sampler,
        discount_factor: float,
        coverage_sampler: Optional[types.Sampler] = None,
    ):
        self.state_sampler = state_sampler
        self.action_sampler = action_sampler
        # self.coverage_sampler = coverage_sampler or self.build_coverage_sampler()
        self.discount_factor = discount_factor

    # def build_coverage_sampler(self):
    #     """Creates a coverage sampler from state and action samplers.
    #
    #     If no coverage sampler is provided, we assume that the coverage distribution
    #     is the product distribution from (state, action, next_state) according
    #     to the distributions induced by the state and action samplers.
    #     """
    #     state_sample = self.state_sampler.sample()
    #     action_sample = self.action_sampler.sample()
    #     next_state_sample = self.state_sampler.sample()
    #     # take cartesian product of

    def canonicalize(self, x: types.RewardFunction, /) -> types.RewardFunction:
        def canonical_reward_fn(state, action, next_state, /):
            state_sample = self.state_sampler.sample()
            action_sample = self.action_sampler.sample()
            next_state_sample = self.state_sampler.sample()
            # ``state``, ``action``, ``next_state`` correspond to s, a, s' in the paper
            # and ``state_sample``, ``action_sample``, ``next_state_sample``
            # correspond to S, A, S' in the paper.
            x_kw = utils.keywordize_rew_fn_call(x)  # call x using keyword arguments
            x_cart = utils.cartesian_call_wrapper(x_kw)  # automatic cartesian product

            # E[R(s', A, S')]. We sample action and next state,
            # and pass in ``next_state`` as the state.
            # We make this the first dimension to then take the mean.
            term_1 = np.mean(
                x_cart(
                    {"action": action_sample, "next_state": next_state_sample},
                    {"state": next_state},
                ),
                axis=0,
            )
            # E[R(s, A, S')]. We also sample action and next state.
            # Now it's simply ``state`` that we pass in as the state.
            term_2 = np.mean(
                x_cart(
                    {"action": action_sample, "next_state": next_state_sample},
                    {"state": state},
                ),
                axis=0,
            )
            # E[R(S, A, S')]. We sample state, action, and next state.
            # This does not require cartesian product over batches
            # as it's not a random variable.
            term_3 = np.mean(x(state_sample, action_sample, next_state_sample), axis=0)

            return (
                x(state, action, next_state)
                + self.discount_factor * term_1
                - term_2
                - self.discount_factor * term_3
            )

        return canonical_reward_fn

    def distance(
        self, x_canonical: types.RewardFunction, y_canonical: types.RewardFunction, /
    ) -> float:
        state_sample = self.state_sampler.sample()
        action_sample = self.action_sampler.sample()
        next_state_sample = self.state_sampler.sample()

        x_samples = x_canonical(state_sample, action_sample, next_state_sample)
        y_samples = y_canonical(state_sample, action_sample, next_state_sample)

        return np.sqrt(1 - np.corrcoef(x_samples, y_samples))

    def epic(self, x: types.RewardFunction, y: types.RewardFunction, /) -> float:
        x_canonical = self.canonicalize(x)
        y_canonical = self.canonicalize(y)
        return self.distance(x_canonical, y_canonical)
