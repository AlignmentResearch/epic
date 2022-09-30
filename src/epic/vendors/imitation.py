from epic import types
from imitation.rewards import reward_nets


def reward_net_to_fn(reward_net: reward_nets.RewardNet) -> types.RewardFunction:
    """Converts an imitation reward net to a reward function."""

    def reward_fn(state, action, next_state, done):
        return (
            reward_net.forward(state, action, next_state, done).cpu().detach().numpy()
        )

    return reward_fn
