import gym
import numpy as np
import pytest

from epic.distances import divergence_free
from epic.samplers import DummyGymStateSampler, GymSpaceSampler


def rew_fn_0(state, action, next_state, _):
    return np.zeros_like(state)


def rew_fn_0_potential_shaping(state, action, next_state, _):
    return 2.0 * next_state - 3.0 * state


def rew_fn_1(state, action, next_state, _):
    return state + next_state + action


def rew_fn_2(state, action, next_state, _):
    return -(state**0.5) - next_state**2 - 6 * np.log(action + 1.0)


def rew_fn_1_potential_shaping(state, action, next_state, _):
    return state + next_state + action + next_state / 4 - state / 7


def test_divergence_free_dist_no_errors():
    state_space = gym.spaces.Discrete(10)
    action_space = gym.spaces.Discrete(10)

    x = rew_fn_1
    y = rew_fn_2

    dist = divergence_free.DivergenceFree(
        state_sampler=DummyGymStateSampler(space=state_space),
        action_sampler=GymSpaceSampler(space=action_space),
        discount_factor=1,
    ).distance(x, y, n_samples_cov=500, n_samples_can=500)

    print(dist)

    assert isinstance(dist, float)
    assert not np.isnan(dist)


def test_divergence_free_dist_reward_equivalence():
    state_space = gym.spaces.Discrete(10)
    action_space = gym.spaces.Discrete(10)

    x = rew_fn_1
    y = rew_fn_1_potential_shaping

    dist = divergence_free.DivergenceFree(
        state_sampler=DummyGymStateSampler(space=state_space),
        action_sampler=GymSpaceSampler(space=action_space),
        discount_factor=1,
    ).distance(x, y, n_samples_cov=5000, n_samples_can=25000)

    print(dist)

    assert np.isclose(dist, 0, atol=5e-2)
