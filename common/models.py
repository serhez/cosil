import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class EnsembleQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(EnsembleQNetwork, self).__init__()

        n_nets = 2

        self.nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_inputs + num_actions, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(n_nets)
            ]
        )

    def min(self, state, action):
        xu = torch.cat([state, action], -1)
        return torch.min(torch.stack([net(xu) for net in self.nets]), 0)[0]

    def forward(self, state, action):
        xu = torch.cat([state, action], -1)
        return [net(xu) for net in self.nets]


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], -1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class Discriminator(nn.Module):
    def __init__(self, n_obs) -> None:
        super().__init__()

        n_hidden = 128

        self.net = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_obs, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, state):
        return self.net(state)


class MorphoValueFunction(nn.Module):
    def __init__(self, n_morpho) -> None:
        super().__init__()
        """
        n_hidden = 64

        self.net = nn.Sequential(
            nn.Linear(n_morpho, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )
        """
        self.net = nn.Linear(n_morpho * 3, 1)

    def forward(self, state):
        feats = torch.cat((state, state**2, state**3), -1)
        return self.net(feats)


class WassersteinCritic(nn.Module):
    def __init__(self, n_obs, obs_normalizers=None) -> None:
        super().__init__()

        self.obs_normalizers = obs_normalizers

        n_hidden = 200

        self.net = nn.Sequential(
            nn.Linear(n_obs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, state):
        if self.obs_normalizers is not None:
            state = (state - self.obs_normalizers[0]) / (self.obs_normalizers[1] + 1e-6)
        return self.net(state)


class InverseDynamics(nn.Module):
    def __init__(self, n_obs, n_actions, action_space=None) -> None:
        super().__init__()

        n_hidden = 256

        self.net = nn.Sequential(
            nn.Linear(n_obs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions),
            nn.Tanh(),
        )

        # action rescaling
        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            self.register_buffer(
                "action_scale",
                torch.FloatTensor((action_space.high - action_space.low) / 2.0),
            )
            self.register_buffer(
                "action_bias",
                torch.FloatTensor((action_space.high + action_space.low) / 2.0),
            )

    def forward(self, state, next_state, morpho_params):
        xu = torch.cat([state, next_state, morpho_params], -1)
        y_t = self.net(xu)
        action = y_t * self.action_scale + self.action_bias
        return action


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean, normal

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0.0, std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.0), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
