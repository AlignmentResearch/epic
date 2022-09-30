import abc

from epic import types


class Distance(abc.ABC):
    def __init__(
        self,
        state_sampler: types.StateSampler,
        action_sampler: types.ActionSampler,
        discount_factor: float,
    ):
        self.state_sampler = state_sampler
        self.action_sampler = action_sampler
        self.discount_factor = discount_factor

    @abc.abstractmethod
    def canonicalize(
        self,
        reward_function: types.RewardFunction,
        /,
        n_samples_cov: int,
        n_samples_can: int,
    ) -> types.RewardFunction:
        """Canonicalize a reward function.

        Args:
            reward_function: The reward function to canonicalize.
            n_samples_cov: The number of samples to draw from the coverage distribution.
            n_samples_can: The number of samples to draw for the canonicalization step
                for each sample of the coverage distribution. The total number of
                samples drawn is ``n_samples_cov * n_samples_can``.

        Returns:
            The canonicalized reward function.
        """

    @abc.abstractmethod
    def _distance(
        self,
        x_canonical: types.RewardFunction,
        y_canonical: types.RewardFunction,
        /,
        n_samples_cov: int,
        n_samples_can: int,
    ) -> float:
        """Subclass to implement the distance computation between two canonicalized
        reward functions.

        Args:
            x_canonical: The first canonicalized reward function.
            y_canonical: The second canonicalized reward function.

        Returns:
            The distance between the two reward functions.
        """

    def distance(
        self,
        x: types.RewardFunction,
        y: types.RewardFunction,
        /,
        n_samples_cov: int,
        n_samples_can: int,
    ) -> float:
        """Compute the distance between two reward functions.

        Args:
            x: The first reward function.
            y: The second reward function.
            n_samples_cov: The number of samples to draw from the coverage distribution.
            n_samples_can: The number of samples to draw for the canonicalization step
                for each sample of the coverage distribution. The total number of
                samples drawn is ``n_samples_cov * n_samples_can``.

        Returns:
            The distance between the two reward functions.
        """
        x_canonical = self.canonicalize(x, n_samples_cov, n_samples_can)
        y_canonical = self.canonicalize(y, n_samples_cov, n_samples_can)
        return self._distance(x_canonical, y_canonical, n_samples_cov, n_samples_can)
