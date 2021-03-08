import gym
import cv2
import matplotlib.pyplot as plt
from gym.wrappers import GrayScaleObservation, FrameStack

from car_racing.normalize_observation import NormalizeObservation


class Environment(gym.Wrapper):
    def __init__(self, env_name):
        env = gym.make(env_name)

        self.env = GrayScaleObservation(env)
        self.env = NormalizeObservation(self.env)
        self.env = FrameStack(self.env, 4)

        gym.Wrapper.__init__(self, self.env)
    # def reset(self):
    #     state = self.env.reset()
    #     # state_g = self._image_to_gray(state)
    #     return state
    #
    # def step(self, action):
    #     next_state, reward, done, info = self.env.step(action)
    #     next_state_g = self._image_to_gray(next_state)
    #
    #     return next_state_g, reward, done, info
    #
    # @staticmethod
    # def _image_to_gray(state):
    #     return cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

    # @staticmethod
    # def plot_state(state):
    #     plt.imshow(state, cmap='gray', vmin=0, vmax=255)
    #     plt.show()
