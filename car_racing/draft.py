import numpy as np
from tensorboardX import SummaryWriter
import gym
import torch
from torch import nn
from torch.nn import functional as F
from collections import namedtuple

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done', 'next_state'])

GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 128


def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(nn.Linear(input_size, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, n_actions)
                                 )

    def forward(self, x):
        return self.net(x)


def unpack_batch_a2c(batch, net, last_val_gamma, device="cpu"):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = torch.FloatTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
        last_vals_v = net(last_states_v)[2]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v


if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-reinforce")

    net = PGN(env.observation_space.shape[0], env.action_space.n)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    done_episodes = 0
    total_rewards = []
    batch = []
    batch_episodes = 0

    batch_states, batch_actions, batch_qvals = [], [], []
    rewards = []
    EPISODES = 10000

    for e in range(EPISODES):
        state = env.reset()
        total_reward = 0
        while True:
            env.render()

            state_v = torch.Tensor(state)
            actions_v = net(state_v)
            actions_softmax = F.softmax(actions_v, dim=-1).data.numpy()

            action = np.random.choice([0, 1], p=actions_softmax)
            next_state, reward, done, _ = env.step(action)

            exp = Experience(state, action, reward, done, next_state)
            batch.append(exp)

            if done:
                total_reward = np.sum(rewards)
                rewards.clear()
                break

            state = next_state

        if len(batch) < BATCH_SIZE:
            continue

        unpack_batch(batch, net)
        optimizer.zero_grad()
        batch_states_v = torch.FloatTensor(batch_states)
        batch_actions_v = torch.LongTensor(batch_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals)

        y_hat = net(batch_states_v)
        log_actions = F.log_softmax(y_hat, dim=1)
        # Minimize the logarithm of the probability of the action a given state S times the return R
        log_prob_actions_v = batch_qvals_v * log_actions[range(len(batch_states)), batch_actions_v]
        loss = -log_prob_actions_v.mean()

        loss.backward()
        optimizer.step()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()



