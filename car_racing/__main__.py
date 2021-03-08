import numpy as np

from car_racing.environment import Environment
from car_racing.model import ActorCritic
from car_racing.agent import Agent, Experience
from car_racing.config import config, hyper_parameters

if __name__ == "__main__":

    env = Environment(config.environment)
    net = ActorCritic(env.observation_space.shape, env.action_space.shape[0])
    agent = Agent(env, net, hyper_parameters)

    training_score = 0
    batch = []

    for e in range(config.num_episodes):
        state = env.reset()
        episode_score = 0
        while True:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)

            exp = Experience(state, action, reward, next_state)
            batch.append(exp)

            if len(batch) > config.batch_size:
                print('Updating parameters...')
                agent.train(batch)
                batch.clear()

            episode_score += reward

            if done:
                break

            state = next_state

        training_score = 0.9 * training_score + 0.1 * episode_score

        if e % 100 == 0:
            print(f'Ep: {e}: Last score: {np.round(episode_score, 2)}, Average score: {np.round(training_score, 2)} ')

        if training_score > env.spec.reward_threshold:
            print("Training completed - problem solved!")
            break
