from typing import Dict, Protocol, Sequence, TypedDict

import numpy as np
from typing_extensions import ParamSpec

from epic import types


def keywordize_rew_fn(reward_fn: types.RewardFunction):
    def wrapper(*, state: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        return reward_fn(state, action, next_state)

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


def product_batch_call(
    function: Fn, *grouped_arguments: Dict[str, np.ndarray]
) -> np.ndarray:
    arg_names = [
        arg_name
        for arg_group in grouped_arguments
        for arg_name in list(arg_group.keys())
    ]
    assert len(arg_names) == len(
        set(arg_names)
    ), "Argument names must not be unique within and across groups"

    batch_lengths = []
    for arg_group in grouped_arguments:
        shapes = [arg_val.shape[0] for arg_val in arg_group.values()]
        assert (
            len(set(shapes)) == 1
        ), "All arrays in the same axis must have the same number of batch items"
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


def product_batch_wrapper(function: Fn):
    def wrapper(*args: Dict[str, np.ndarray]):
        return product_batch_call(function, *args)

    return wrapper
