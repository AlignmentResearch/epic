import gym
import numpy as np

from epic import samplers, types
from epic.distances import divergence_free
import einops


def rew_fn_0(state, action, next_state, _):
    return einops.reduce(np.zeros(state.shape[0]), "batch ... -> batch", "sum")


def rew_fn_0_potential_shaping(state, action, next_state, _):
    return einops.reduce((2.0 * next_state - 2.0 * state), "batch ... -> batch", "sum")


def rew_fn_1(state, action, next_state, _):
    return state + next_state + action


def rew_fn_1_potential_shaping(state, action, next_state, _):
    return state + next_state + action + next_state / 3 - state / 3


def rew_fn_2(state, action, next_state, _):
    return action + np.log(next_state + 1) - state**2


def rew_fn_2_potential_shaping(state, action, next_state, _):
    return action + np.log(next_state + 1) - state**2 + 3 * np.sqrt(next_state) - 4 * np.sqrt(state)


def test_divergence_free_dist_no_errors():
    state_space = gym.spaces.Discrete(10)
    action_space = gym.spaces.Discrete(10)

    x = rew_fn_1
    y = rew_fn_2

    dist = divergence_free.DivergenceFree(
        coverage_sampler=samplers.ProductDistrCoverageSampler(
            samplers.GymSpaceSampler(space=action_space),
            samplers.DummyGymStateSampler(space=state_space),
        ),
        training_hyperparams=types.PotentialTrainingHyperparams(batch_size=100),
        discount_factor=1,
    ).distance(x, y, n_samples_cov=100, n_samples_can=100)

    assert isinstance(dist, float)
    assert not np.isnan(dist)


def test_divergence_free_dist_reward_equivalence_constant_reward():
    state_space = gym.spaces.Discrete(10)
    action_space = gym.spaces.Discrete(10)

    x = rew_fn_0
    y = rew_fn_0_potential_shaping

    dist = divergence_free.DivergenceFree(
        coverage_sampler=samplers.ProductDistrCoverageSampler(
            samplers.GymSpaceSampler(space=action_space),
            samplers.DummyGymStateSampler(space=state_space),
        ),
        training_hyperparams=types.PotentialTrainingHyperparams(batch_size=2500),
        discount_factor=1,
    ).distance(x, y, n_samples_cov=500, n_samples_can=2500)

    print(dist)

    assert np.isclose(dist, 0, atol=1e-7)


def test_divergence_free_dist_reward_equivalence_constant_reward_multiple_dims():
    state_space = gym.spaces.MultiDiscrete([10, 10])
    action_space = gym.spaces.MultiDiscrete([10, 10])

    x = rew_fn_0
    y = rew_fn_0_potential_shaping

    dist = divergence_free.DivergenceFree(
        coverage_sampler=samplers.ProductDistrCoverageSampler(
            samplers.GymSpaceSampler(space=action_space),
            samplers.DummyGymStateSampler(space=state_space),
        ),
        training_hyperparams=types.PotentialTrainingHyperparams(batch_size=2500),
        discount_factor=1,
    ).distance(x, y, n_samples_cov=500, n_samples_can=2500)

    print(dist)

    assert np.isclose(dist, 0, atol=1e-7)


def test_divergence_free_dist_reward_equivalence_constant_reward_multiple_dims_continuous():
    state_space = gym.spaces.Box(low=-10, high=10, shape=(10,))
    action_space = gym.spaces.Box(low=-10, high=10, shape=(10,))

    x = rew_fn_0
    y = rew_fn_0_potential_shaping

    dist = divergence_free.DivergenceFree(
        coverage_sampler=samplers.ProductDistrCoverageSampler(
            samplers.GymSpaceSampler(space=action_space),
            samplers.DummyGymStateSampler(space=state_space),
        ),
        training_hyperparams=types.PotentialTrainingHyperparams(batch_size=10000),
        discount_factor=1,
    ).distance(x, y, n_samples_cov=500, n_samples_can=10000)

    print(dist)

    assert np.isclose(dist, 0, atol=1e-7)


def test_divergence_free_dist_reward_equivalence_linear_reward():
    state_space = gym.spaces.Discrete(10)
    action_space = gym.spaces.Discrete(10)

    x = rew_fn_1
    y = rew_fn_1_potential_shaping

    dist = divergence_free.DivergenceFree(
        coverage_sampler=samplers.ProductDistrCoverageSampler(
            samplers.GymSpaceSampler(space=action_space),
            samplers.DummyGymStateSampler(space=state_space),
        ),
        training_hyperparams=types.PotentialTrainingHyperparams(batch_size=300000),
        discount_factor=1,
    ).distance(x, y, n_samples_cov=10000, n_samples_can=300000)

    print(dist)

    assert np.isclose(dist, 0, atol=2e-2)


def test_divergence_free_dist_reward_equivalence_complex_reward():
    state_space = gym.spaces.Discrete(10)
    action_space = gym.spaces.Discrete(10)

    x = rew_fn_2
    y = rew_fn_2_potential_shaping

    dist = divergence_free.DivergenceFree(
        coverage_sampler=samplers.ProductDistrCoverageSampler(
            samplers.GymSpaceSampler(space=action_space),
            samplers.DummyGymStateSampler(space=state_space),
        ),
        training_hyperparams=types.PotentialTrainingHyperparams(batch_size=400000),
        discount_factor=0.75,
    ).distance(x, y, n_samples_cov=10000, n_samples_can=400000)

    print(dist)

    assert np.isclose(dist, 0, atol=5e-2)
