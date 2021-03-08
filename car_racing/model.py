import numpy as np
import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, state_shape, n_actions):
        super(ActorCritic, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(state_shape)
        self.policy_mean = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Tanh(),
        )

        self.policy_variance = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softplus(),
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # fx = x.float() / 256
        fx = x.float()
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy_mean(conv_out), self.policy_variance(conv_out), self.value(conv_out)
