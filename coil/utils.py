# Based on https://github.com/pranz24/pytorch-soft-actor-critic (MIT Licensed)

import math
import time
from types import SimpleNamespace
import torch
from torch import device, nn
import gym
import numpy as np
import itertools
from torch.nn.functional import softplus

import matplotlib.pyplot as plt
from matplotlib import gridspec
import pyswarms as ps
from model import Discriminator, WassersteinCritic
from sklearn.decomposition import PCA
import ot
from replay_memory import ReplayMemory
import copy

from run_gpy import create_GPBO_model, get_new_candidates_BO
import cma

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def dict_add(target, new_data):
    for key, val in new_data.items():
        if key in target:
            if isinstance(val, dict):
                dict_add(target[key], val)
            elif hasattr(val, "__add__"):
                target[key] += val
            else:
                # Unknown data type,just leave it
                pass
        else:
            #target[key] = copy.deepcopy(val)
            target[key] = val

def dict_div(target, value):
    for key, val in target.items():
        if isinstance(val, dict):
            dict_div(target[key], value)
        elif hasattr(val, "__truediv__"):
            target[key] /= value
        else:
            # Unknown data type,just leave it
            pass

class ObservationsRecorderWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self._env = env
        self.obs_buffer = []
        self.dones = []

    def step(self, action):
        state, reward, terminated, truncated, info = self._env.step(action)
        self.obs_buffer.append(info)
        self.dones.append(terminated or truncated)
        return state, reward, terminated, truncated, info

    def get_stacked_dict(self):
        res = {}
        for o in self.obs_buffer:
            for k, v in o.items():
                if k not in res:
                    res[k] = []
                res[k].append(v)
        for k, v in res.items():
            res[k] = np.stack(v)
        
        res['dones'] = np.array(self.dones)

        return res

def get_marker_info(info_dict, legs, marker_idx, *, pos_type="norm", vel_type="rel", torso_type=None, head_type=None, head_wrt=None):

    all_data = []
    all_keys = []

    for l, m in itertools.product(legs, marker_idx):
        if pos_type and pos_type != "skip":
            # If None/False/skip is passed, skip positions altogether
            pos_key = f"track/{pos_type}/pos/l{l}/m{m}"
            all_keys.append(pos_key)
            data = info_dict[pos_key]
            all_data.append(data)
            
        if vel_type and vel_type != "skip":
            # If None/False/skip is passed, skip velocities altogether
            vel_key = f"track/{vel_type}/vel/l{l}/m{m}"
            all_keys.append(vel_key)
            data = info_dict[vel_key]
            all_data.append(data)
    
    if torso_type is not None:
        for t_type in torso_type:
            torso_key = f"track/abs/{t_type}/torso"
            all_keys.append(torso_key)
            data = info_dict[torso_key]
            all_data.append(data)
    
    if head_type is not None:
        for h_type in head_type:
            for h_wrt in head_wrt:
                head_key = f"track/norm/{h_type}/head_wrt_{h_wrt}"
                all_keys.append(head_key)
                data = info_dict[head_key]
                all_data.append(data)

    all_data = np.concatenate(all_data, axis=-1)

    if 'dones' in info_dict:
        eps = []
        i = 0
        for j, d in enumerate(info_dict['dones']):
            if d:
                eps.append(all_data[i:j+1])
                i = j + 1
        return eps, all_keys
        
    return all_data, all_keys

# TODO: Study
def train_wgan_critic(opt, critic, expert_obs: list, memory, batch_size=256, use_transitions=True):

        opt.zero_grad()
        device = expert_obs[0].device

        episode_lengths = [len(ep) for ep in expert_obs]
        correct_inds = []
        len_sum = 0
        for length in episode_lengths:
            correct_inds.append(torch.arange(length - 1) + len_sum)
            len_sum += length

        correct_inds = torch.cat(correct_inds)

        expert_obs = torch.cat(expert_obs, dim=0)
        expert_inds = correct_inds[torch.randint(0, len(correct_inds), (batch_size, ))]
        expert_feats = expert_obs[expert_inds]

        if use_transitions:
            expert_feats = torch.cat((expert_feats, expert_obs[expert_inds + 1]), dim=1) 

        _, _, _, _, _, marker_samples, next_marker_samples = memory.sample(batch_size)
        marker_feats = marker_samples
        
        if use_transitions:
            marker_feats = np.concatenate((marker_feats, next_marker_samples), axis=1)

        marker_samples = torch.as_tensor(marker_feats).float().to(device)

        assert expert_feats.shape == marker_samples.shape


        expert_scores = critic(expert_feats)
        policy_scores = critic(marker_samples)

        alpha = torch.rand(marker_samples.shape[0], 1, device=device)

        interpolates = alpha * expert_feats + ((1 - alpha) * marker_samples)
        interpolates.requires_grad_()
        interp_scores = critic(interpolates)

        lamb = 10
        gradients = torch.autograd.grad(
            outputs=interp_scores,
            inputs=interpolates,
            grad_outputs=torch.ones(len(interpolates), 1, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        loss = (policy_scores - expert_scores).mean() + lamb * gradient_penalty

        loss.backward()
        opt.step()

        return loss.item(), expert_scores.detach().mean().item(), policy_scores.detach().mean().item(), gradient_penalty.item()

# TODO: Study
def train_disc(opt, disc: Discriminator, expert_obs: torch.Tensor, memory, use_transitions=False):

        opt.zero_grad()
        disc.train()

        disc_loss = nn.BCEWithLogitsLoss()

        batch_size = 64
        device = expert_obs[0].device

        episode_lengths = [len(ep) for ep in expert_obs]
        correct_inds = []
        len_sum = 0
        for length in episode_lengths:
            correct_inds.append(torch.arange(length - 1) + len_sum)
            len_sum += length

        correct_inds = torch.cat(correct_inds)

        expert_obs = torch.cat(expert_obs, dim=0)
        expert_inds = correct_inds[torch.randint(0, len(correct_inds), (batch_size, ))]
        expert_feats = expert_obs[expert_inds]

        if use_transitions:
            expert_feats = torch.cat((expert_feats, expert_obs[expert_inds + 1]), dim=1) 

        _, _, _, _, _, marker_samples, next_marker_samples = memory.sample(batch_size)
        marker_feats = marker_samples

        if use_transitions:
            marker_feats = np.concatenate((marker_feats, next_marker_samples), axis=1)

        marker_samples = torch.as_tensor(marker_feats).float().to(device)

        assert expert_feats.shape == marker_samples.shape

        expert_disc = disc(expert_feats)
        policy_disc = disc(marker_samples)

        expert_labels = 0.8*torch.ones(len(expert_feats), 1, device=expert_obs.device) # NOTE: Shouldn't expert labels be all ones?
        policy_labels = 0.2*torch.ones(len(marker_feats), 1, device=expert_obs.device) # NOTE: Shouldn't policy labels be all zeros?
        # NOTE: This is so the discriminator learns to distinguish between expert and policy samples

        loss = disc_loss(expert_disc, expert_labels) \
            + disc_loss(policy_disc, policy_labels)

        loss.backward()
        opt.step()

        return loss.item(), expert_disc.sigmoid().detach().mean().item(), policy_disc.sigmoid().detach().mean().item()

def visualize_morphos(morphos, optimized_or_not):
    X = np.stack(morphos)
    op = np.stack(optimized_or_not)

    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)

    fig = plt.figure(dpi=100, figsize=(8,8))
    
    plt.scatter(X_transformed[op, 0], X_transformed[op, 1], c='green', label='Optimized')
    plt.scatter(X_transformed[~op, 0], X_transformed[~op, 1], c='red', label='Exploration')
    plt.legend()
    return fig

def obj(morpho_params, cheetah_lengths_torch, initial_states_torch, agent, evaluate_grads=False):
    if not type(morpho_params) == torch.Tensor:
        morpho_params = torch.as_tensor(morpho_params, device=initial_states_torch.device, dtype=torch.float32)

    states_torch = initial_states_torch.clone()

    if evaluate_grads:
        morpho_params.requires_grad_()

    if len(morpho_params.shape) == 2:
        states_torch = states_torch.repeat(len(morpho_params), 1, 1)
        morpho_params = morpho_params.view(len(morpho_params), 1, -1)
    states_torch[..., agent.morpho_slice] = morpho_params

    _, _, action, _ = agent.policy.sample(states_torch)

    q_val = agent.critic.min(states_torch, action)

    loss = -q_val.mean(-1).mean(-1)

    if evaluate_grads:
        loss.mean().backward()
        print('Morpho grads')
        print(morpho_params.grad)
        return loss, morpho_params.grad.abs().sum().item()

    return loss

@torch.no_grad()
def plot_full_q_fn(fn, bounds, current_optima, real_lengths=None):

    optima = torch.as_tensor(current_optima).float()
    optima = optima.repeat(100, 1)

    fig = plt.figure(figsize=(12,12), dpi=100)

    for i in range(len(bounds)):
        x = torch.linspace(bounds[i, 0], bounds[i, 1], 100).float()
        param_to_test = optima.clone()
        param_to_test[:, i] = x
        losses = fn(param_to_test.to('cuda:0'))
        assert losses.shape == (100, )
        plt.subplot(2,3, i+1)
        plt.axvline(current_optima[i], c='black', linestyle='--')
        if real_lengths and len(real_lengths) > i:
            plt.axvline(real_lengths[i], c='green', linestyle='--')
        plt.plot(x, losses)
        #plt.xlabel(f'Scale parameter {i+1}')
        #plt.ylabel('Negative value')

    return fig

def obj_morpho_value(morpho_params, agent,):
    if not type(morpho_params) == torch.Tensor:
        morpho_params = torch.as_tensor(morpho_params, device=agent.device, dtype=torch.float32)

    loss = agent.morpho_value(morpho_params).mean(-1)

    return loss

def optimize_morpho_params_pso(agent, initial_states, bounds, replay, use_distance_value=False, device='cpu'):
    initial_states_torch = torch.as_tensor(np.stack(initial_states), dtype=torch.float32, device=device)
    #initial_states_torch = torch.as_tensor(replay.sample(1024)[0], dtype=torch.float32, device=device)

    # Subsample initial states
    n_samples = 1024
    #initial_states_torch = initial_states_torch[torch.randint(0, len(initial_states_torch), (n_samples, ))]
    initial_states_torch = initial_states_torch[-n_samples:]
    
    @torch.no_grad()
    def fn(x):
        if use_distance_value:
            losses = obj_morpho_value(x, agent,).cpu().numpy()
        else:
            losses = obj(x, None, initial_states_torch, agent).cpu().numpy()
        
        assert losses.shape == (x.shape[0], )
        torch.cuda.synchronize()
        return losses

    #bounds = (np.array([0.5]), np.array([2.]))
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    optimizer = ps.single.GlobalBestPSO(n_particles=250, dimensions=bounds.shape[0], options=options, bounds=(bounds[:,0].numpy(), bounds[:, 1].numpy()), ftol=1e-7, ftol_iter=30)
    cost, pos = optimizer.optimize(fn, iters=250)

    # print gradients. They should be zero-ish if we are at local optimum
    #_, grads_abs_sum = obj(pos, None, initial_states_torch, policy, q_function, evaluate_grads=True)

    fig = plot_full_q_fn(fn, bounds, pos)
    morpho_params = torch.tensor(pos)

    return cost, morpho_params, fig, 0

def merge_batches(first_batch, second_batch):
    return [np.concatenate([a, b], axis=0) for a, b in zip(first_batch, second_batch)]

def _distance(training_transitions, demo):
    cost_train_distance = ot.dist(training_transitions, demo)
    distance = ot.emd2([], [], cost_train_distance, numItermax=1000000, numThreads="max")
    del cost_train_distance

    return distance

def compute_distance(training_transitions, demo, to_match):
    # Compute distances separately for velocities and positions
    pos_indices = []
    idx = 0
    for key in to_match:
        if "pos" in key:
            pos_indices += [idx, idx+1, idx+2]
        idx += 3 

    pos_distance = _distance(training_transitions[:, pos_indices], demo[:, pos_indices])

    vel_indices = []
    idx = 0
    for key in to_match:
        if "vel" in key:
            vel_indices += [idx, idx+1, idx+2]
        idx += 3 

    vel_distance = _distance(training_transitions[:, vel_indices], demo[:, vel_indices])
    
    return pos_distance, vel_distance

def handle_absorbing(feats, action, reward, next_feats, mask, marker_obs, next_marker_obs, memory, obs_size, pwil_rewarder=None):
    
    marker_obs = np.concatenate([marker_obs, np.zeros(1)])
    if feats.shape[0] != obs_size:
        feats = np.concatenate([feats, np.zeros(1)])
    
    if mask == 0.:
        # next obs is absorbing
        next_marker_obs = np.zeros(marker_obs.shape[0])
        next_marker_obs[-1] = 1.
        next_feats = np.zeros_like(feats)
        next_feats[-1] = 1.
        
        # in addition add transition from absorbing to itself
        add_action = np.zeros_like(action)
        absorbing_state = next_marker_obs.copy()
        memory.push(next_feats, add_action, 0., next_feats, 1., absorbing_state, absorbing_state)
    else:
        next_feats = np.concatenate([next_feats, np.zeros(1)])
        next_marker_obs = np.concatenate([next_marker_obs, np.zeros(1)])
    
    if pwil_rewarder is not None:    
        pwil_reward = pwil_rewarder.compute_reward({'observation': next_marker_obs})
        reward = pwil_reward

    memory.push(feats, action, reward, next_feats, 1., marker_obs, next_marker_obs)


def create_replay_data(env, memory, marker_info_fn, agent, absorbing_state=True, steps=5000):

    start_time = time.time()
    step = 0
    while step < steps:
        state, _ = env.reset()
        marker_obs, _ = marker_info_fn(env.get_track_dict())
        done = False
        rsum = 0

        while not done:
            feats = np.concatenate([state, env.morpho_params])
            if absorbing_state:
                feats = np.concatenate([feats, np.zeros(1)])

            action = agent.select_action(feats, evaluate=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rsum += info["reward_run"]
            
            next_marker_obs, _ = marker_info_fn(info)
            next_feats = np.concatenate([next_state, env.morpho_params])

            mask = 1.

            if absorbing_state:
                handle_absorbing(feats, action, reward, next_feats, mask, marker_obs, next_marker_obs, memory, agent.num_inputs + agent.num_morpho_obs)
            else:
                memory.push(feats, action, reward, next_feats, mask, marker_obs, next_marker_obs)

            state = next_state
            marker_obs = next_marker_obs

            step += 1
        print('Episode done', rsum)
    
    print(f'Took {time.time() - start_time:.2f}')

def bo_step(args, morphos, num_morpho, pos_train_distances, env):
    bo_args = SimpleNamespace()
    bo_args.mean = args.bo_gp_mean
    bo_args.kernel = "Matern52"
    bo_args.optimizer = "lbfgsb"
    bo_args.gp_type = "GPR"
    bo_args.acq_type = "LCB"
    bo_args.add_bias = 0
    bo_args.add_linear=0

    # TODO change when resuming from pretrained
    prev_morphos_to_consider = 200
    if args.env_name == "GaitTrackHalfCheetah-v0":
        prev_morphos_to_consider = 200
    
    X = np.array(morphos).reshape(-1, args.episodes_per_morpho, num_morpho)[:, 0][-prev_morphos_to_consider:]
    Y = np.array(pos_train_distances).reshape(-1, args.episodes_per_morpho).mean(1, keepdims=True)[-prev_morphos_to_consider:]
    
    model = create_GPBO_model(bo_args, X, Y)
    x_next, x_exploit_next, bo_step = get_new_candidates_BO(model, X, Y, args.env_name, env.min_task, env.max_task, args.acq_weight, None, acq_type="LCB")
    morpho_params_np = x_next.flatten()
    optimized_morpho_params = x_exploit_next.flatten()
    print("Exploit morpho params", optimized_morpho_params)

    return morpho_params_np, optimized_morpho_params

def rs_step(args, num_morpho, morphos, pos_train_distances, min_task, max_task):
    
    # Average over same morphologies
    X = np.array(morphos).reshape(-1, args.episodes_per_morpho, num_morpho)[:, 0]
    Y = np.array(pos_train_distances).reshape(-1, args.episodes_per_morpho).mean(1, keepdims=True)
    
    curr = X[-1]
    if Y[-1] > Y[-2]:
        curr = X[-2]

    curr += np.random.normal(size=curr.shape) * 0.05
    curr = np.clip(curr, min_task, max_task)
    print('RS: new params', curr)

    # Second is always best found so far
    return curr, X[np.argmin(Y)]
