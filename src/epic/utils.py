from typing import Dict, Protocol, TypedDict

import numpy as np
from typing_extensions import ParamSpec

from epic import types


def keywordize_rew_fn(reward_fn: types.RewardFunction):
    def wrapper(
        *,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ):
        return reward_fn(state, action, next_state, done)

    return wrapper


P = ParamSpec("P")


class Fn(Protocol[P]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> np.ndarray:
        ...


def multidim_batch_call(
    function: Fn, arguments: Dict[str, np.ndarray], num_batch_dims: int
):
    """Allows converting a function that only processes calls with a single batch dimension
    into a function that processes calls with multiple batch dimensions.

    This is done by flattening the arguments into a single batch dimension, and then
    unflattening (reshaping) the results.
    """
    assert num_batch_dims > 0
    flattened_arguments = {}
    for key, value in arguments.items():
        flattened_arguments[key] = value.reshape(-1, *value.shape[num_batch_dims:])
    result = function(**flattened_arguments)
    return result.reshape(
        *next(iter(arguments.values())).shape[:num_batch_dims], *result.shape[1:]
    )


def multidim_rew_fn(function: types.RewardFunction):
    function = keywordize_rew_fn(function)

    def wrapper(state, action, next_state, done, batch_dims: int = 1):
        return multidim_batch_call(
            function,
            dict(state=state, action=action, next_state=next_state, done=done),
            num_batch_dims=batch_dims,
        )

    return wrapper


def product_batch_call(
    function: Fn, *grouped_arguments: Dict[str, np.ndarray]
) -> np.ndarray:
    arg_names = [
        arg_name
        for arg_group in grouped_arguments
        for arg_name in list(arg_group.keys())
    ]
    if len(arg_names) != len(set(arg_names)):
        raise ValueError("Argument names must be unique within and across groups")

    batch_lengths = []
    for arg_group in grouped_arguments:
        shapes = [arg_val.shape[0] for arg_val in arg_group.values()]
        if len(set(shapes)) != 1:
            raise ValueError(
                "All arrays in the same axis must have the same number of batch items"
            )
        batch_lengths.append(shapes[0])

    class Info(TypedDict):
        axis: int
        array: np.ndarray

    map_arg_to_info: Dict[str, Info] = dict()
    for index, arg_group in enumerate(grouped_arguments):
        for arg_name, arg_val in arg_group.items():
            map_arg_to_info[arg_name] = {"axis": index, "array": arg_val}

    arguments: Dict[str, np.ndarray] = dict()
    for arg_name, arg_info in map_arg_to_info.items():
        # for each axis i that is not data['axis'], we want to copy data['array']
        # along the axis i for batch_lengths[i] times
        axis = arg_info["axis"]
        array = arg_info["array"]
        batch_lengths_sliced = [
            length for idx, length in enumerate(batch_lengths) if idx != axis
        ]
        tiled_array = np.broadcast_to(array, (*batch_lengths_sliced, *array.shape))
        # now we want to swap the axes, since we want the first dimension of `array`
        # (the original batch axis) to be along the axis `axis`. The other axes should
        # be in the correct order since we first excluded it on `batch_lengths_sliced`.
        tiled_array = np.swapaxes(tiled_array, -len(array.shape), axis)
        # this is how the shape of the array should be:
        assert tiled_array.shape == tuple((*batch_lengths, *array.shape[1:]))
        arguments[arg_name] = tiled_array

    return multidim_batch_call(
        function, arguments=arguments, num_batch_dims=len(batch_lengths)
    )


def product_batch_wrapper(function: Fn, nested=True):
    def wrapper(*args: Dict[str, np.ndarray]):
        if nested:
            return product_batch_call(function, *args)
        else:
            # merge all arguments into a single dictionary
            merged_args = {}
            for arg_group in args:
                merged_args.update(arg_group)
            return function(**merged_args)

    return wrapper


def broadcast(state, action, next_state, done, /, n_samples_can):
    """Tiles the state, action, next_state, and done arrays along the first axis"""
    return (
        np.swapaxes(np.broadcast_to(state, (n_samples_can, *state.shape)), 1, 0),
        np.swapaxes(np.broadcast_to(action, (n_samples_can, *action.shape)), 1, 0),
        np.swapaxes(
            np.broadcast_to(next_state, (n_samples_can, *next_state.shape)), 1, 0
        ),
        np.swapaxes(np.broadcast_to(done, (n_samples_can, *done.shape)), 1, 0),
    )


def reshape(state, action, next_state, done, /, n_samples_cov):
    """Reshapes the state, action, next_state, and done arrays to have shape (n_samples_cov, -1)"""
    return (
        state.reshape(n_samples_cov, -1),
        action.reshape(n_samples_cov, -1),
        next_state.reshape(n_samples_cov, -1),
        done.reshape(n_samples_cov, -1),
    )
