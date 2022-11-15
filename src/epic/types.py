import abc
from typing import Protocol, Callable, Union

import numpy as np
import numpy.typing as npt

# import torch

# from epic import utils


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


# class PotentialShapingFunction(Protocol):
#     """Abstract class for a potential shaping function."""

#     def __call__(
#         self,
#         state: Union[npt.NDArray, torch.Tensor],
#         next_state: Union[npt.NDArray, torch.Tensor],
#         potential: Callable,
#         discount_factor: float,
#         /,
#         return_tensor: bool = False,
#     ) -> Union[npt.NDArray, torch.Tensor]:
#         """Compute potential shaping function outputs for a batch of state transitions.
#         Args:
#             state: Current states of shape `(batch_size,) + state_shape`.
#             next_state: Successor states of shape `(batch_size,) + state_shape`.
#             potential: The potential function. Responsibility is on caller to ensure
#             that it will accept the types of state and next_state.
#             discount_factor: The discount applied to the next state.
#             return_tensor: Whether to return a PyTorch Tensor or a numpy nd.array.
#         """
#         if isinstance(state, torch.Tensor):
#             assert isinstance(next_state, torch.Tensor), "State and Next State are not of the same type."
#             if return_tensor:
#                 return (discount_factor * potential(next_state) - potential(state)).reshape(-1)
#             else:
#                 return utils.numpy_from_tensor(discount_factor * potential(next_state) - potential(state)).reshape(-1)
#         else:
#             assert isinstance(next_state, np.ndarray), "State and Next State are not of the same type."
#             if return_tensor:
#                 return utils.float_tensor_from_numpy(
#                     discount_factor * potential(next_state) - potential(state)
#                 ).reshape(-1)
#             else:
#                 return (discount_factor * potential(next_state) - potential(state)).reshape(-1)
