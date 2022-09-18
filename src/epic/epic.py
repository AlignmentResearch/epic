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
    ):
        self.state_sampler = state_sampler
        self.action_sampler = action_sampler
        self.discount_factor = discount_factor

    def canonicalize(self, x: types.RewardFunction, /) -> types.RewardFunction:
        def canonical_reward_fn(state, action, next_state, /):
            state_sample = self.state_sampler.sample()
            action_sample = self.action_sampler.sample()
            next_state_sample = self.state_sampler.sample()
            # ``state``, ``action``, ``next_state`` correspond to s, a, s' in the paper
            # and ``state_sample``, ``action_sample``, ``next_state_sample``
            # correspond to S, A, S' in the paper.
            x_kw = utils.keywordize_rew_fn(x)  # call x using keyword arguments
            # automatically take the cartesian product of independent batches.
            x_cart = utils.product_batch_wrapper(x_kw)

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

    def _distance(
        self, x_canonical: types.RewardFunction, y_canonical: types.RewardFunction, /
    ) -> float:
        state_sample = self.state_sampler.sample()
        action_sample = self.action_sampler.sample()
        next_state_sample = self.state_sampler.sample()

        x_samples = x_canonical(state_sample, action_sample, next_state_sample)
        y_samples = y_canonical(state_sample, action_sample, next_state_sample)

        return np.sqrt(1 - np.corrcoef(x_samples, y_samples)[0, 1])

    def distance(self, x: types.RewardFunction, y: types.RewardFunction, /) -> float:
        x_canonical = self.canonicalize(x)
        y_canonical = self.canonicalize(y)
        return self._distance(x_canonical, y_canonical)


def epic_distance(x, y, /, *, state_sampler, action_sampler, discount_factor):
    return EPIC(
        state_sampler=state_sampler,
        action_sampler=action_sampler,
        discount_factor=discount_factor,
    ).distance(x, y)
