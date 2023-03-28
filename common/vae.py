# From https://github.com/FangchenLiu/SAIL (MIT Licensed)

from collections import deque
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

MAX_LOG_STD = 0.5
MIN_LOG_STD = -20

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_size=256):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mean = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=MIN_LOG_STD, max=MAX_LOG_STD)
        return mean, log_std


class Decoder(torch.nn.Module):
    def __init__(self, input_dim, out_dim, hidden_size=256):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class VAE(torch.nn.Module):
    def __init__(self, state_dim, hidden_size=128, latent_dim=64):
        super(VAE, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = Encoder(state_dim, latent_dim=latent_dim, hidden_size=self.hidden_size)
        self.decoder = Decoder(latent_dim, state_dim, hidden_size=self.hidden_size)

    def forward(self, state):
        mu, log_sigma = self.encoder(state)
        sigma = torch.exp(log_sigma)
        sample = mu + torch.randn_like(mu)*sigma
        self.z_mean = mu
        self.z_sigma = sigma

        return self.decoder(sample)

    def get_next_states(self, states):
        mu, _ = self.encoder(states)
        return self.decoder(mu)

    def get_loss(self, state, next_state):
        next_pred = self.get_next_states(state)
        return ((next_state-next_pred)**2).mean()

    def train(self, expert_obs, epoch, optimizer, batch_size=128, beta=0.2):
        
        if len(expert_obs) > 1:
            test_ep = expert_obs[-1]
            train_eps = expert_obs[:-1]
        else:
            test_ep = expert_obs[0]
            train_eps = expert_obs[0:1]

        episode_lengths = [len(ep) for ep in train_eps]
        correct_inds = []
        len_sum = 0
        for length in episode_lengths:
            correct_inds.append(torch.arange(length - 1) + len_sum)
            len_sum += length

        correct_inds = torch.cat(correct_inds)

        num_batch = int(np.ceil(len(correct_inds) / batch_size))
        input = torch.cat(train_eps, dim=0)
        print(input.shape)

        deq_len = 500
        prev_val_losses = deque(maxlen=deq_len)

        mean_loss = 0
        for epoch in range(epoch):
            mean_loss = 0
            all_indices = correct_inds[torch.randperm(len(correct_inds))]

            for batch_num in range(num_batch-1):
                batch_idxs = all_indices[batch_num:batch_num + batch_size]

                train_in = input[batch_idxs].float()
                train_targ = input[batch_idxs + 1].float()
                optimizer.zero_grad()
                dec = self.forward(train_in)
                reconstruct_loss = ((train_targ-dec)**2).mean()
                ll = latent_loss(self.z_mean, self.z_sigma)
                loss = reconstruct_loss + beta*ll
                loss.backward()
                
                mean_loss += loss.item()

                optimizer.step()
            
            mean_loss /= num_batch
            val_dec = self.get_next_states(test_ep[:-1])
            val_loss = ((test_ep[1:]-val_dec)**2).mean()
            
            print(f'Epoch {epoch} loss {mean_loss:.5f} val {val_loss:.5f}')

            if epoch > 1000 and val_loss > (sum(prev_val_losses) / deq_len):
                print('Early stopping VAE training due to increasing loss')
                break

            prev_val_losses.append(val_loss.item())

        return mean_loss
