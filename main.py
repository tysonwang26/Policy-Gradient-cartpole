
import gym
import RL
from draw_graph import Plot
import time

env = gym.make('CartPole-v0')
env = env.unwrapped

plot_ = Plot()

MAX_EPISODES = 2000
MAX_STEP_EPISODES = 5000

RL = RL.PolicyGradient(n_actions=env.action_space.n,
                       n_features=env.observation_space.shape[0],
                       n_hidden=10,
                       learning_rate=0.01,
                       reward_decay=0.99,
                       epsilon=0.90
                       )

def main():
    for i in range(1, MAX_EPISODES):
        print(i, "of episodes", end="\n")
        start_time = time.time()
        observation = env.reset()
        for j in range(MAX_STEP_EPISODES):
            env.render()
            action = RL.choose_action(observation)
            if j < 5:
                action=0
            observation_, reward, done, info = env.step(action)
            RL.store_transition(observation, action, reward, False)

            if done:
                RL.store_transition(observation, action, 0.0, True)
                RL.learn()
                break
            observation = observation_

        end_time = time.time()
        plot_.plot_graph((end_time - start_time), i)
    env.close()
    RL.store_net()

if __name__ == '__main__':
    main()
