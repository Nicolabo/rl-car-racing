import numpy as np

from gym.spaces import Box
from gym import ObservationWrapper


class NormalizeObservation(ObservationWrapper):
    r"""Normalize already grayscale image observation."""

    def __init__(self, env):
        super(NormalizeObservation, self).__init__(env)

        obs_shape = self.observation_space.shape
        self.observation_space = Box(low=0, high=1, shape=(obs_shape[0], obs_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return observation / 256
