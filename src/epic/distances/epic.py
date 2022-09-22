import numpy as np

from epic import types, utils
from epic.distances import base


class EPIC(base.Distance):
    def canonicalize(
        self, x: types.RewardFunction, /, nested=True
    ) -> types.RewardFunction:
        """Canonicalize a reward function.

        Applies the canonically shaped reward transformation defined in 4.1 of
        https://arxiv.org/pdf/2006.13900.pdf.

        Args:
            x: The reward function to canonicalize.
            nested: Whether sampling is nested over any expectation operators, i.e.
                take the cartesian product of the state,action,next state samples
                provided when the reward function is called with the samples used
                to compute the expectation. If not using nested sampling, the
                batch size when calling the canonicalized reward function must be the
                same as the batch size of the samples used to compute the expectation.


        """

        def canonical_reward_fn(state, action, next_state, /):
            state_sample = self.state_sampler.sample()
            action_sample = self.action_sampler.sample()
            next_state_sample = self.state_sampler.sample()
            assert (
                state_sample.shape[0]
                == action_sample.shape[0]
                == next_state_sample.shape[0]
            )
            assert state.shape[0] == action.shape[0] == next_state.shape[0]
            if not nested:
                assert state.shape[0] == state_sample.shape[0]

            # ``state``, ``action``, ``next_state`` correspond to s, a, s' in the paper
            # and ``state_sample``, ``action_sample``, ``next_state_sample``
            # correspond to S, A, S' in the paper.
            x_kw = utils.keywordize_rew_fn(x)  # call x using keyword arguments
            # automatically take the cartesian product of independent batches.
            x_cart = utils.product_batch_wrapper(x_kw, nested=nested)

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


def epic_distance(x, y, /, *, state_sampler, action_sampler, discount_factor, nested):
    """Compute the EPIC distance between two reward functions.

    Helper for automatically instantiating the EPIC Distance class and computing
    the distance between two reward functions.

    Do not ue this helper if you want to compute distances between multiple pairs
    of reward functions. Instead, instantiate the EPIC class and call the
    ``distance`` method on the instance multiple times.

    Args:
        x: The first reward function.
        y: The second reward function.
        state_sampler: A sampler for the state space.
        action_sampler: A sampler for the action space.
        discount_factor: The discount factor of the MDP.
        nested: Whether sampling is nested over any expectation operators. See
            ``EPIC.canonicalize`` for more details.

    Returns:
        The EPIC distance between the two reward functions.
    """
    return EPIC(
        state_sampler=state_sampler,
        action_sampler=action_sampler,
        discount_factor=discount_factor,
    ).distance(x, y, nested=nested)
