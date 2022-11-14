import gym
import numpy as np
import pytest

from epic.distances import divergence_free
from epic.samplers import DummyGymStateSampler, GymSpaceSampler
from einops import reduce


def rew_fn_0(state, action, next_state, _):
    return np.zeros(state.shape[0])


def rew_fn_0_potential_shaping(state, action, next_state, _):
    return reduce((2.0 * next_state - 2.0 * state), "b ... -> b", "sum")


def rew_fn_1(state, action, next_state, _):
    return state + next_state + action


def rew_fn_1_potential_shaping(state, action, next_state, _):
    return state + next_state + action + next_state / 3 - state / 3


def rew_fn_2(state, action, next_state, _):
    return state**2 + next_state**2 + 2 * action


def test_divergence_free_dist_no_errors():
    state_space = gym.spaces.Discrete(10)
    action_space = gym.spaces.Discrete(10)

    x = rew_fn_1
    y = rew_fn_2

    dist = divergence_free.DivergenceFree(
        state_sampler=DummyGymStateSampler(space=state_space),
        action_sampler=GymSpaceSampler(space=action_space),
        discount_factor=1,
    ).distance(x, y, n_samples_cov=500, n_samples_can=1000)

    assert isinstance(dist, float)
    assert not np.isnan(dist)


def test_divergence_free_dist_reward_equivalence_constant_reward():
    state_space = gym.spaces.Discrete(10)
    action_space = gym.spaces.Discrete(10)

    x = rew_fn_0
    y = rew_fn_0_potential_shaping

    dist = divergence_free.DivergenceFree(
        state_sampler=DummyGymStateSampler(space=state_space),
        action_sampler=GymSpaceSampler(space=action_space),
        discount_factor=1,
    ).distance(x, y, n_samples_cov=500, n_samples_can=2500)

    print(dist)

    assert np.isclose(dist, 0, atol=1e-7)


def test_divergence_free_dist_reward_equivalence_linear_reward():
    state_space = gym.spaces.Discrete(10)
    action_space = gym.spaces.Discrete(10)

    x = rew_fn_1
    y = rew_fn_1_potential_shaping

    dist = divergence_free.DivergenceFree(
        state_sampler=DummyGymStateSampler(space=state_space),
        action_sampler=GymSpaceSampler(space=action_space),
        discount_factor=1,
    ).distance(x, y, n_samples_cov=500, n_samples_can=5000)

    print(dist)

    assert np.isclose(dist, 0, atol=2e-1)
