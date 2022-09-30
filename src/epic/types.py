import abc
from typing import Protocol, Union, Generic, TypeVar, Tuple

import numpy.typing as npt
import numpy as np


class RewardFunction(Protocol):
    """Abstract class for reward function.
    Requires implementation of __call__() to compute the reward given a batch of
    states, actions, and next states.
    """

    def __call__(
        self,
        state: npt.NDArray,
        action: npt.NDArray,
        next_state: npt.NDArray,
        done: npt.NDArray[np.bool_],
        /,
    ) -> npt.NDArray:
        """Compute rewards for a batch of transitions.
        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
        Returns:
            Computed rewards of shape `(batch_size,`).
        """


class CoverageDistribution(Protocol):
    """Abstract class for coverage distribution."""

    def __call__(
        self,
        state: npt.NDArray,
        action: npt.NDArray,
        next_state: npt.NDArray,
        done: npt.NDArray[np.bool_],
        /,
    ) -> npt.NDArray:
        """Compute coverage for a batch of transitions.
        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
        Returns:
            Probability for (s,a,s') triple of shape `(batch_size,`).
        """


Coverage = Union[
    CoverageDistribution,
    # CoverageGrid
]


T_co = TypeVar("T_co", covariant=True)


class Sampler(Protocol[T_co]):
    def sample(self, n_samples: int) -> T_co:
        pass


class ActionSampler(Sampler[npt.NDArray]):
    pass


class StateSampler(Sampler[Tuple[npt.NDArray[np.bool_], npt.NDArray]]):
    pass
