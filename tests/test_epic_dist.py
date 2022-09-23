import numpy as np
import pytest
from epic.distances import epic
from epic.samplers import GymSpaceSampler, DummyGymStateSampler
import gym


def rew_fn_1(state, action, next_state, _):
    return state + next_state + action


def rew_fn_2(state, action, next_state, _):
    return state**2 + next_state**2 + 2 * action


def rew_fn_1_potential_shaping(state, action, next_state, _):
    return state + next_state + action + next_state / 3 - state / 3


def test_epic_dist_no_errors():
    state_space = gym.spaces.Discrete(10)
    action_space = gym.spaces.Discrete(10)

    x = rew_fn_1
    y = rew_fn_2

    dist = epic.EPIC(
        state_sampler=DummyGymStateSampler(space=state_space, n_samples=100),
        action_sampler=GymSpaceSampler(space=action_space, n_samples=100),
        discount_factor=1,
    ).distance(x, y)

    assert isinstance(dist, float)


def test_epic_dist_reward_equivalence():
    state_space = gym.spaces.Discrete(10)
    action_space = gym.spaces.Discrete(10)

    x = rew_fn_1
    y = rew_fn_1_potential_shaping

    dist = epic.EPIC(
        state_sampler=DummyGymStateSampler(space=state_space, n_samples=100),
        action_sampler=GymSpaceSampler(space=action_space, n_samples=100),
        discount_factor=1,
    ).distance(x, y)

    assert np.isclose(dist, 0, atol=1e-7)


def test_epic_dist_no_nested():
    state_space = gym.spaces.Discrete(10)
    action_space = gym.spaces.Discrete(10)

    x = rew_fn_1
    y = rew_fn_2

    dist = epic.EPIC(
        state_sampler=DummyGymStateSampler(space=state_space, n_samples=1000),
        action_sampler=GymSpaceSampler(space=action_space, n_samples=1000),
        discount_factor=1,
    ).distance(x, y, nested=False)

    print(dist)

    assert isinstance(dist, float)
