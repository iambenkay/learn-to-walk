import gym
import numpy as np
from agent import Agent
import matplotlib.pyplot as plt


def ddpg_algorithm(load_checkpoint):
    env = gym.make('BipedalWalker-v3', render_mode='human')
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])

    n_games = 1000

    best_score = env.reward_range[0]
    score_history = []

    if load_checkpoint:
        n_steps = 0

        while n_steps <= agent.batch_size:
            observation, _ = env.reset()
            env.render()
            action = env.action_space.sample()
            new_observation, reward, done, *info = env.step(action)
            agent.remember(observation, action, reward, new_observation, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action = agent.choose_action(observation, evaluate)
            new_observation, reward, done, *info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, new_observation, done)

            if not load_checkpoint:
                agent.learn()
            observation = new_observation
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print('episode ', i, 'score: %.1f' % score, 'avg score: %.1f' % avg_score)
    if not load_checkpoint:
        x = [i for i in range(n_games)]
        plot_learning_curve(x, score_history, 'learning_curve.png')


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
