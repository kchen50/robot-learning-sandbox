import torch
import torch.nn as nn

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim * 2)

    @torch.no_grad()
    def sample_action(self, state, noise=False):
        x = torch.relu(self.fc1(state))
        value = self.fc2(x)
        cov = torch.diag(0.01 * torch.sigmoid(value[0][2:]))
        mean = value[0][:2]
        mean = torch.tanh(mean) * 0.01
        dist = torch.distributions.MultivariateNormal(mean, cov)
        return dist.sample(), cov[0][0]

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        value = self.fc2(x)
        cov = torch.diag(
            0.01 * torch.sigmoid(value[0][2:])
        )  #torch.diag(torch.exp(value[0][2:]))
        # cov = torch.tanh(cov)
        mean = value[0][:2]
        mean = torch.tanh(mean) * 0.01
        dist = torch.distributions.MultivariateNormal(mean, cov)
        return dist.log_prob(action)


class Critic(nn.Module):

    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 1)  # Value function (single output)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        value = self.fc2(x)
        return value