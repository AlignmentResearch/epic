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
        self, x: types.RewardFunction, /, nested=True
    ) -> types.RewardFunction:
        """Canonicalize a reward function.

        Args:
            x: The reward function to canonicalize.
            nested: Whether sampling is nested over any expectation operators, i.e.
                take the cartesian product of the state,action,next state samples
                provided when the reward function is called with the samples used
                to compute the expectation. If not using nested sampling, the
                batch size when calling the canonicalized reward function must be the
                same as the batch size of the samples used to compute the expectation.

        Returns:
            The canonicalized reward function.
        """

    @abc.abstractmethod
    def _distance(
        self, x_canonical: types.RewardFunction, y_canonical: types.RewardFunction, /
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
        self, x: types.RewardFunction, y: types.RewardFunction, /, nested=True
    ) -> float:
        """Compute the distance between two reward functions.

        Args:
            x: The first reward function.
            y: The second reward function.
            nested: Whether sampling is nested over any expectation operators when
                computing the canonicalization. See ``canonicalize`` for more details.

        Returns:
            The distance between the two reward functions.
        """
        x_canonical = self.canonicalize(x, nested=nested)
        y_canonical = self.canonicalize(y, nested=nested)
        return self._distance(x_canonical, y_canonical)
