import math

from collections import namedtuple
import numpy as np
import torch
from torch import optim
from torch.distributions import Normal
import torch.nn.functional as F
from typing import List, Tuple

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'last_state'])


class Agent:
    def __init__(self, env, net, device, hyper_parameters):
        self._env = env
        self._net = net
        self._device = device
        self._hyper_parameters = hyper_parameters

        self._optimizer = optim.Adam(self._net.parameters(), lr=hyper_parameters.learning_rate)

    def take_action(self, state: np.array) -> List[float]:
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            mu, sigma, _ = self._net(state)

        dist = Normal(mu, sigma)
        action_values = dist.sample()
        action = action_values.squeeze().numpy()
        clipped_action = self._clip_action(action)
        return clipped_action

    def train(self, batch):
        states, actions, rewards, last_states = self._unpack_batch(batch)

        states_v = torch.FloatTensor(np.array(states, copy=False))
        actions_v = torch.FloatTensor(actions)

        mu_v, var_v, value_v = self._net(states_v)

        critic_loss, advantage = self._critic_update(rewards=rewards,
                                                     last_states=last_states,
                                                     predict_values=value_v)

        log_probs = advantage * self._get_log_probability(mu_v, var_v, actions_v)
        actor_loss = -log_probs.mean()

        # entropy = -(torch.log(2 * math.pi * var_v) + 1) / 2
        # entropy_loss = self._config.entropy_beta * ent_v.mean()
        # loss = actor_loss + entropy_loss + critic_loss

        loss = actor_loss + critic_loss

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def _clip_action(self, action: np.array) -> List[float]:
        action_low = self._env.action_space.low
        action_high = self._env.action_space.high
        return [np.clip(a, lt, ht) for a, lt, ht in zip(action, action_low, action_high)]

    @staticmethod
    def _unpack_batch(batch):
        states = []
        actions = []
        rewards = []
        last_states = []
        for exp in batch:
            states.append(np.array(exp.state, copy=False))
            actions.append(exp.action)
            rewards.append(exp.reward)
            last_states.append(np.array(exp.last_state, copy=False))

        return states, actions, rewards, last_states

    def _critic_update(self,
                       rewards: List[int],
                       last_states: List[np.array],
                       predict_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards_np = np.array(rewards)

        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(self._device)
        last_vals = self._net(last_states_v)[2]
        last_vals_np = last_vals.squeeze().data.numpy()
        last_vals_np *= self._hyper_parameters.gamma ** self._hyper_parameters.reward_steps
        rewards_np += last_vals_np

        return_v = torch.FloatTensor(rewards_np).to(self._device)

        loss = F.mse_loss(predict_values.squeeze(-1), return_v)
        advantage = return_v.unsqueeze(dim=-1) - predict_values.detach()

        return loss, advantage

    def _actor_update(self):
        pass

    @staticmethod
    def _get_log_probability(mu_v, var_v, actions_v):
        p1 = - ((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
        return p1 + p2
