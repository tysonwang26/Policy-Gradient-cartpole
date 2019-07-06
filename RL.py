import torch
import torch.nn.functional as F
import numpy as np


class Net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.hidden.weight.data.normal_(0, 0.1)  # initialization
        self.output = torch.nn.Linear(n_hidden, n_output)
        self.output.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = F.relu(self.hidden(x))
        output = F.softmax(self.output(x), 0)
        return output

class PolicyGradient:
    def __init__(self,
                 n_actions,
                 n_features,
                 n_hidden=10,
                 learning_rate=0.01,
                 reward_decay=0.95,
                 epsilon=0.99
                 ):

        self.n_actions = n_actions
        self.n_features = n_features
        self.reward_decay = reward_decay
        self.epsilon = epsilon

        # list of stored data
        self.ep_obs, self.ep_acts, self.ep_rews = [], [], []

        # Set net
        if self.load_net():
            self.net = Net(self.n_features, n_hidden, self.n_actions)
        self.net.cuda()
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
        self.loss_fun = torch.nn.MSELoss()

    def choose_action(self, observation):
        if np.random.uniform() < self.epsilon:
            prob_actions = self.net(torch.FloatTensor(observation).cuda())
            action = torch.argmax(prob_actions).cpu().data.numpy()
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, observation, action, reward, done):
        if done:
            self.ep_rews[len(self.ep_rews) - 1] = reward
        else:
            self.ep_obs.append(observation)
            self.ep_acts.append(action)
            self.ep_rews.append(reward)

    def learn(self):
        # get all prob of stored actions
        discounted_rewards = torch.FloatTensor(self._discount_rewards())
        ep_observation = torch.FloatTensor(np.asarray(self.ep_obs))

        self.optimizer.zero_grad()
        for i in range(len(ep_observation)):
            state = ep_observation[i]
            reward = discounted_rewards[i]

            prob_actions = self.net(torch.FloatTensor(state).cuda())
            max_of_prob = torch.max(prob_actions).cpu()

            loss = torch.sum(reward * torch.log(max_of_prob) * -1, -1)
            loss.backward()

        self.optimizer.step()

        self.ep_obs, self.ep_acts, self.ep_rews = [], [], []

        return None

    def _discount_rewards(self):
        discount_ep_reward = np.zeros_like(self.ep_rews)
        run = 0.0
        for i in reversed(range(len(discount_ep_reward))):
            run = run * self.reward_decay + self.ep_rews[i]
            discount_ep_reward[i] = run

        discount_ep_reward -= np.mean(discount_ep_reward)
        discount_ep_reward /= np.std(discount_ep_reward)
        return discount_ep_reward

    def store_net(self):
        try:
            torch.save(self.net, 'policy_gradient_net.pkl')
            print("<<Store net complete>>")
        except:
            print("<<Store net error>>")

    def load_net(self):
        try:
            self.net = torch.load('policy_gradient_net.pkl')
            print("<<Load net complete>>")
            return False
        except:
            print("<<Load net error or no model>>")
            return True
