# Based on https://github.com/pranz24/pytorch-soft-actor-critic (MIT Licensed)

import argparse
from collections import deque
import os
import gym
from gait_track_envs import register_env
import numpy as np
import itertools
import torch
from model import WassersteinCritic, Discriminator
from sac import SAC
from replay_memory import ReplayMemory
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from utils import bo_step, dict_add, dict_div, get_marker_info, rs_step, train_disc, optimize_morpho_params_pso, train_wgan_critic, compute_distance, handle_absorbing
from torch import initial_seed, nn, optim
import time
import wandb
import glob
import os
import random
from rewarder import PWILRewarder
from run_gpy import create_GPBO_model, get_new_candidates_BO
import cma

def single_run(args: argparse.Namespace):

    register_env(args.env_name)
    env = gym.make(args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        device = "cuda"
    else:
        device = "cpu"

    # Bounds for morphology optimization
    highs = torch.tensor(env.max_task)
    lows = torch.tensor(env.min_task)
    bounds = torch.stack([lows, highs], dim=1)

    # The distribution used for morphology exploration
    morpho_dist = torch.distributions.Uniform(lows, highs)

    # Is the current morpho optimized or random?
    optimized_morpho = True
    if args.fixed_morpho is not None:
        print('Fixing morphology to', args.fixed_morpho)
        env.set_task(*args.fixed_morpho)
    
    if args.co_adapt:
        morpho_params = morpho_dist.sample().numpy()
        env.set_task(*morpho_params)
        optimized_morpho = False
    
    morpho_params_np = np.array(env.morpho_params)
    # Number of morphology parameters
    num_morpho = env.morpho_params.shape[0]

    memory = ReplayMemory(args.replay_size, args.seed)
    initial_states_memory = []

    metrics = {
        "reward": [],
        "vel_test": [],
        "pos_test": []
    }

    total_numsteps = 0
    updates = 0

    expert_legs = args.expert_legs
    policy_legs = args.policy_legs
    expert_limb_indices = args.expert_markers
    policy_limb_indices = args.policy_markers

    if args.expert_env_name is not None:
        expert_env = gym.make(args.expert_env_name)
        expert_lengths = expert_env.limb_segment_lengths

    # Load CMU or mujoco-generated demos
    if os.path.isdir(args.expert_demos):
        expert_obs = []
        for filepath in glob.iglob(f'{args.expert_demos}/expert_cmu_{args.subject_id}*.pt'):
            episode = torch.load(filepath)
            episode_obs_np, to_match = get_marker_info(episode, expert_legs, expert_limb_indices,
                pos_type=args.pos_type, vel_type=args.vel_type, torso_type=args.torso_type, head_type=args.head_type, head_wrt=args.head_wrt)
            episode_obs = torch.from_numpy(episode_obs_np).float().to(device)
            expert_obs.append(episode_obs)
    else:
        expert_obs = torch.load(args.expert_demos)
        expert_obs_np, to_match = get_marker_info(expert_obs, expert_legs, expert_limb_indices,
                pos_type=args.pos_type, vel_type=args.vel_type, torso_type=args.torso_type, head_type=args.head_type, head_wrt=args.head_wrt)

        expert_obs = [torch.from_numpy(x).float().to(device) for x in expert_obs_np]
        print(f'Expert obs {len(expert_obs)} episodes loaded')
    
    # For terminating environments like Humanoid it is important to use absorbing state
    # From paper Discriminator-actor-critic: Addressing sample inefficiency and reward bias in adversarial imitation learning
    if args.absorbing_state:
        print('Adding absorbing states')
        expert_obs = [torch.cat([ep, torch.zeros(ep.size(0), 1, device=device)], dim=-1) for ep in expert_obs]
        
    obs_size = env.observation_space.shape[0]
    if args.absorbing_state:
        obs_size += 1

    # The dimensionality of each state in demo (marker state)
    demo_dim = expert_obs[0].shape[-1]

    agent = SAC(obs_size, env.action_space, num_morpho, demo_dim, len(env.morpho_params), args)

    # If training the discriminator on transitions, it becomes (s, s')
    if args.learn_disc_transitions:
        demo_dim *= 2

    print("Keys to match:", to_match)
    print("Expert observation shapes:", [x.shape for x in expert_obs])

    if args.algo == 'GAIL':
        disc = Discriminator(demo_dim).to(device)
        disc_opt = optim.AdamW(disc.parameters(), lr=1e-4, weight_decay=1)
    elif args.algo == 'SAIL':
        normalizers = None
        if args.normalize_obs:
            normalizers = (torch.cat(expert_obs).mean(0, keepdim=True), torch.cat(expert_obs).std(0, keepdim=True))
        disc = WassersteinCritic(demo_dim, normalizers).to(device)
        disc_opt = optim.Adam(disc.parameters(), lr=3e-4, betas=(0.5, 0.9), weight_decay=args.disc_weight_decay)
        
        # SAIL includes a pretraining step for the VAE and inverse dynamics
        vae_loss = agent.pretrain_vae(expert_obs, epochs=10000, batch_size=args.batch_size)
        if not args.resume:
            g_inv_loss = agent.run_random_and_pretrain_g(env, lambda x: get_marker_info(x, policy_legs, policy_limb_indices,
                        pos_type=args.pos_type, vel_type=args.vel_type, torso_type=args.torso_type, head_type=args.head_type, head_wrt=args.head_wrt), morpho_dist, expert_obs, co_adapt=args.co_adapt)
    elif args.algo == 'PWIL':
        pass
    else:
        raise NotImplementedError

    if args.resume is not None:
        agent.load_checkpoint(disc, disc_opt, memory, args.resume)
        print(f'Loaded {args.resume}')
        print('Loaded', len(memory), 'transitions')

    morphos = []
    distances = []
    pos_train_distances = []
    optimized_or_not = [False]

    # Compute the mean distance between expert demonstrations
    # This is "demonstrations" in the paper plots
    pos_baseline_distance = 0
    vel_baseline_distance = 0

    if len(expert_obs) > 1:
        num_comp = 0
        for i in range(len(expert_obs)):
            for j in range(len(expert_obs)):
                # W(x, y) = W(y, x), so there's no need to calculate both
                if j >= i:
                    continue
                ep_a = expert_obs[i].cpu().numpy()
                ep_b = expert_obs[j].cpu().numpy()

                pos_dist, vel_dist = compute_distance(ep_a, ep_b, to_match)
                pos_baseline_distance += pos_dist
                vel_baseline_distance += vel_dist
                num_comp += 1
        
        pos_baseline_distance /= num_comp
        vel_baseline_distance /= num_comp
        
    # For linear annealing of exploration in Q-function variant
    epsilon = 1.

    prev_best_reward = -99

    # We experimented with Primal wasserstein imitation learning (Dadaishi et al. 2020) 
    # but did not include experiments in paper as it did not perform well
    if args.algo == 'PWIL':
        pwil_rewarder = PWILRewarder(
            expert_obs,
            False,
            demo_dim,
            num_demonstrations=len(expert_obs),
            time_horizon=300.,
            alpha=5.,
            beta=5.,
            observation_only=True)
    
    # Morphology optimization via distribution distance (for ablations, main results use BO)
    if args.dist_optimizer == 'CMA':
        cma_options = cma.evolution_strategy.CMAOptions()
        cma_options['popsize'] = 5
        cma_options['bounds'] = [0, 1]
        es = cma.CMAEvolutionStrategy([0.5] * len(env.morpho_params), 0.5, inopts=cma_options)
        es_buffer = deque()

    # Main loop
    for i_episode in itertools.count(1):
        start = time.time()

        if args.co_adapt:
            env.set_task(*morpho_params_np)

        episode_reward = 0
        episode_steps = 0
        log_dict, logged = {}, 0
        done = False
        state, _ = env.reset()
        # Compute marker state phi(s) in paper
        marker_obs, to_match = get_marker_info(env.get_track_dict(), policy_legs, policy_limb_indices,
                pos_type=args.pos_type, vel_type=args.vel_type, torso_type=args.torso_type, head_type=args.head_type, head_wrt=args.head_wrt)

        # Morphology parameters xi are included in state in the code
        feats = np.concatenate([state, env.morpho_params])
        if args.absorbing_state:
            initial_states_memory.append(np.concatenate([feats, np.zeros(1)]))
        else:
            initial_states_memory.append(feats)

        train_marker_obs_history = []
        disc_loss, expert_probs, policy_probs, gradient_penalty, g_inv_loss, vae_loss = 0, 0, 0, 0, 0, 0
        
        if args.torso_type and args.torso_type != ['vel']:
            x_pos_history = []
            x_pos_index = to_match.index('track/abs/pos/torso') * 3
        
        if args.algo == 'PWIL':
            pwil_rewarder.reset()
            disc = None

        while not done:
            # Algorithm 1 line 5-
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                feats = np.concatenate([state, env.morpho_params])
                if args.absorbing_state:
                    feats = np.concatenate([feats, np.zeros(1)])

                action = agent.select_action(feats)  # Sample action from policy

            if len(memory) > args.batch_size:

                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    if total_numsteps % args.train_every == 0:
                        # Different algo variants discriminator update (pseudocode line 8-9)
                        if args.algo == 'GAIL':
                            disc_loss, expert_probs, policy_probs = train_disc(disc_opt, disc, expert_obs, memory, use_transitions=args.learn_disc_transitions)
                        elif args.algo == 'SAIL':
                            disc_loss, expert_probs, policy_probs, gradient_penalty = train_wgan_critic(disc_opt, disc, expert_obs, memory, batch_size=args.batch_size, use_transitions=args.learn_disc_transitions)
                            g_inv_loss = agent.update_g_inv(memory, batch_size=args.batch_size)
                    # Policy update (pseudocode line 10)
                    if total_numsteps > args.disc_warmup and len(memory) > args.batch_size and (total_numsteps % args.train_every == 0):
                        # Update parameters of all the networks
                        critic_loss, policy_loss, ent_loss, alpha, action_std, mean_modified_reward, entropy, vae_loss, absorbing_reward = agent.update_parameters(memory, expert_obs, args.batch_size, updates, disc, plot_histogram = total_numsteps % 100 == 0)

                        new_log = {'loss/critic_loss': critic_loss,
                                'loss/policy': policy_loss,
                                'loss/policy_prior_loss': vae_loss,
                                'loss/entropy_loss': ent_loss,
                                'loss/disc_loss': disc_loss,
                                'loss/disc_gradient_penalty': gradient_penalty,
                                'loss/g_inv_loss': g_inv_loss,
                                'modified_reward': mean_modified_reward,
                                'absorbing_reward': absorbing_reward,
                                'action_std': action_std,
                                'probs/expert_disc': expert_probs,
                                'probs/policy_disc': policy_probs,
                                'entropy_temperature/alpha': alpha,
                                'entropy_temperature/entropy': entropy
                                }

                        dict_add(log_dict, new_log)
                        logged += 1

                    updates += 1

            # Environment step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # phi(s)
            next_marker_obs, _ = get_marker_info(info, policy_legs, policy_limb_indices, # NOTE: Do we need to get the markers for the next state?
                    pos_type=args.pos_type, vel_type=args.vel_type, torso_type=args.torso_type, head_type=args.head_type, head_wrt=args.head_wrt)
            
            if args.torso_type and args.torso_type != ['vel']:
                x_pos_history.append(next_marker_obs[x_pos_index]) # NOTE: What is this? -> only used for plotting

            train_marker_obs_history.append(marker_obs)

            episode_steps += 1
            total_numsteps += 1
            # Change reward to remove action penalty
            reward = info['reward_run'] # NOTE: Why are we removing the action penalty?
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            # NOTE: Used for handling absorbing states as a hack to get the reward to be 0 when the episode is done, as well as meaning "done" for `memory.push()`
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            
            if args.omit_done:
                mask = 1.

            feats = np.concatenate([state, env.morpho_params])
            next_feats = np.concatenate([next_state, env.morpho_params])

            if args.absorbing_state:
                handle_absorbing(feats, action, reward, next_feats, mask, marker_obs, next_marker_obs, memory, obs_size, pwil_rewarder=(pwil_rewarder if args.algo == 'PWIL' else None))    
            else:
                if args.algo == 'PWIL':
                    reward = pwil_rewarder.compute_reward({'observation': next_marker_obs})
                memory.push(feats, action, reward, next_feats, mask, marker_obs, next_marker_obs)

            state = next_state
            marker_obs = next_marker_obs

            epsilon -= 1. / 1e6

        if total_numsteps > args.num_steps:
            break
        
        # Logging
        dict_div(log_dict, logged)
        s = time.time()
        train_marker_obs_history = np.stack(train_marker_obs_history)

        # Compare Wasserstein distance of episode to all demos
        all_demos = torch.cat(expert_obs).cpu().numpy()
        if args.absorbing_state:
            all_demos = all_demos[:, :-1]

        pos_train_distance, vel_train_distance = compute_distance(train_marker_obs_history, all_demos, to_match)
        train_distance = pos_train_distance + vel_train_distance
        morphos.append(env.morpho_params.flatten())  
        distances.append(train_distance)
        pos_train_distances.append(pos_train_distance)

        if args.save_morphos:
            torch.save({"morphos": morphos, "distances": distances}, 'morphos.pt')

        print(f'Training distance: {train_distance:.2f} - baseline: {(pos_baseline_distance+vel_baseline_distance):.2f} in {time.time()-s:.2f}')
        if args.torso_type and args.torso_type != ['vel']:
            log_dict['xpos'] = wandb.Histogram(np.stack(x_pos_history))
        log_dict['distr_distances/pos_train_distance'] = pos_train_distance
        log_dict['distr_distances/vel_train_distance'] = vel_train_distance
        log_dict['distr_distances/pos_baseline_distance'] = pos_baseline_distance
        log_dict['distr_distances/vel_baseline_distance'] = vel_baseline_distance
        log_dict['episode_steps'] = episode_steps

        if optimized_morpho:
            log_dict["reward_optimized_train"] = episode_reward
        
        optimized_morpho_params = None

        # Adapt morphology.
        # Different variants here based on algorithm used
        # Line 13 in Algorithm 1
        if args.co_adapt and (i_episode % args.episodes_per_morpho == 0):
            if (total_numsteps < args.morpho_warmup):
                print('Sampling morphology')
                morpho_params = morpho_dist.sample()
                morpho_params_np = morpho_params.numpy()
            # Following three use distribution distance morphology adaptation with different optimizers
            # Bayesian optimization (Algorithm 2)
            elif args.dist_optimizer == "BO":
                bo_s = time.time()
                morpho_params_np, optimized_morpho_params = bo_step(args, morphos, num_morpho, pos_train_distances, env)
                optimized_morpho = True
                for j in range(len(morpho_params_np)):
                    log_dict[f'morpho_param_values/morpho_param_{j}'] = morpho_params_np[j]
                for j in range(len(optimized_morpho_params)):
                    log_dict[f'morpho_exploit/morpho_param_{j}'] = optimized_morpho_params[j]
                bo_e = time.time()
                print(f'BO took {bo_e-bo_s:.2f}')
            # Ablation: Random search
            elif args.dist_optimizer == "RS":
                morpho_params_np, optimized_morpho_params = rs_step(args, num_morpho, morphos, pos_train_distances, env.min_task, env.max_task)
            # Ablation: CMA
            elif args.dist_optimizer == "CMA":
                # Average over same morphologies
                X = np.array(morphos).reshape(-1, args.episodes_per_morpho, num_morpho)[:, 0]
                Y = np.array(pos_train_distances).reshape(-1, args.episodes_per_morpho).mean(1, keepdims=True)
                
                if len(es_buffer) == 0:
                    suggestion = es.ask()
                    suggestion = (env.max_task - suggestion) / (env.max_task - env.min_task)
                    
                    [es_buffer.append(m) for m in suggestion]

                    if X.shape[0] >= 5:
                        curr = (X[-5:] - env.min_task) / (env.max_task - env.min_task)
                        es.tell(curr, Y[-5:])

                morpho_params_np = es_buffer.pop()
                optimized_morpho_params = X[np.argmin(Y)]
                
            else:
                # Q-function version
                optimized_morpho = random.random() > epsilon
                
                if (total_numsteps > args.morpho_warmup) and optimized_morpho:
                    print('Optimizing morphology')
                    morpho_loss, morpho_params, fig, grads_abs_sum = optimize_morpho_params_pso(agent, \
                        initial_states_memory, bounds, memory, \
                        use_distance_value=args.train_distance_value, device=device)
                    optimized_morpho_params = morpho_params.clone().numpy()
                    morpho_params_np = morpho_params.detach().numpy()
                    log_dict["morpho/morpho_loss"] = morpho_loss
                    log_dict["morpho/grads_abs_sum"] = grads_abs_sum
                    log_dict['q_fn_scale'] = wandb.Image(fig)

                    for j in range(len(morpho_params_np)):
                        log_dict[f'morpho_param_values/morpho_param_{j}'] = morpho_params_np[j]
                else:
                    print('Sampling morphology')
                    morpho_params = morpho_dist.sample()
                    morpho_params_np = morpho_params.numpy()

            optimized_or_not.append(optimized_morpho)
            # Set new morphology in environment
            env.set_task(*morpho_params_np)

            print('Current morpho')
            print(env.morpho_params)

        log_dict["reward_train"] = episode_reward
        
        if args.save_checkpoints and episode_reward > prev_best_reward:
            ckpt_path = agent.save_checkpoint(disc, disc_opt, memory, args.env_name, "best")
            # These are big so dont save in wandb
            # wandb.save(ckpt_path)
            prev_best_reward = episode_reward
            print('New best reward')

        took = time.time() - start
        log_dict['episode_time'] = took

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {} took: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2), round(took, 3)))

        # Evaluation episodes
        # Also used to make plots
        if i_episode % 20 == 0 and args.eval is True:
            start = time.time()
            test_marker_obs_history = []
            avg_reward = 0.
            avg_steps = 0
            episodes = 10
            if not os.path.exists("videos"):
                os.mkdir("videos")
            vid_path = f"videos/ep_{i_episode}.mp4"
            recorder = VideoRecorder(env, vid_path, enabled=args.record_test)
            
            if args.co_adapt and optimized_morpho_params is not None:
                env.set_task(*optimized_morpho_params)

            for test_ep in range(episodes):
                state, _ = env.reset()
                episode_reward = 0
                done = False
                episode_steps = 0
                if test_ep == 0:
                    recorder.capture_frame()

                while not done:
                    feats = np.concatenate([state, env.morpho_params])
                    if args.absorbing_state:
                        feats = np.concatenate([feats, np.zeros(1)])
                    action = agent.select_action(feats, evaluate=True)

                    next_state, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    if test_ep == 0:
                        recorder.capture_frame()

                    marker_obs, _ = get_marker_info(info, policy_legs, policy_limb_indices,
                            pos_type=args.pos_type, vel_type=args.vel_type, torso_type=args.torso_type, head_type=args.head_type, head_wrt=args.head_wrt)

                    reward = info['reward_run']
                    test_marker_obs_history.append(marker_obs)
                    episode_reward += reward

                    state = next_state
                    episode_steps += 1

                avg_reward += episode_reward
                avg_steps += episode_steps
            avg_reward /= episodes
            avg_steps /= episodes

            log_dict['avg_test_reward'] = avg_reward
            log_dict['avg_test_steps'] = avg_steps
            log_dict["reward_optimized_test"] = avg_reward
            took = time.time() - start

            log_dict['test_time'] = took
            if args.record_test:
                log_dict['test_video'] = wandb.Video(vid_path, fps=20, format="gif")
            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}, Steps: {}, Took {}".format(episodes, round(avg_reward, 2), avg_steps, round(took, 2)))
            print("----------------------------------------")
            if args.save_checkpoints:
                ckpt_path = agent.save_checkpoint(disc, disc_opt, memory, args.env_name, "1")
            # These are big so only save policy on wandb
            # wandb.save(ckpt_path)
            torch.save(agent.policy.state_dict(), 'imitator.pt')
            wandb.save('imitator.pt')

            print('Calculating distributional distance')
            s = time.time()
            # Compute and log distribution distances
            test_marker_obs_history = np.stack(test_marker_obs_history)
            short_exp_demos = torch.cat(expert_obs).cpu().numpy()
            if args.absorbing_state:
                short_exp_demos = short_exp_demos[:, :-1]
            pos_test_distance, vel_test_distance = compute_distance(test_marker_obs_history, short_exp_demos, to_match)
            log_dict["distr_distances/vel_test"] = vel_test_distance
            log_dict["distr_distances/pos_test"] = pos_test_distance

            metrics["vel_test"].append(vel_test_distance)
            metrics["pos_test"].append(pos_test_distance)
            metrics["reward"].append(avg_reward)

            torch.save(metrics, 'metrics.pt')

            print('Took', round(time.time()-s, 2))

            train_marker_obs_history = []
            recorder.close()
        
        log_dict['total_numsteps'] = total_numsteps
        wandb.log(log_dict)
        log_dict, logged = {}, 0

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAIL + SAC + co-adaptation')
    parser.add_argument('--env-name', default='GaitTrackHalfCheetahOriginal-v0',
                        help='Mujoco Gym environment')
    parser.add_argument('--algo', default='GAIL',
                        help='Algorithm GAIL or SAIL or PWIL')                                            
    parser.add_argument('--expert-demos', type=str, default='data/expert_demos_sampled_GaitTrackHalfCheetah-v0.pt',
                        help='Path to the expert demonstration file')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--target_entropy', type=str, default="auto", metavar='G',
                        help='Target value for entropy')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=2000000, metavar='N',
                        help='size of replay buffer (default: 20000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--run-name', default='gail',
                        help='Run name (logging only)')
    parser.add_argument('--log-scale-rewards', action="store_true",
                        help='Use sigmoid directly as reward or log of sigmoid')
    parser.add_argument('--reward-style', type=str, default='GAIL',
                        help='AIRL-style or GAIL-style reward')
    parser.add_argument('--train-every', type=int, default=1,
                        help='Train every N timesteps')
    parser.add_argument('--explore-morpho-episodes', type=int, default=800,
                        help='Episodes to run morphology exploration for')
    parser.add_argument('--morpho-warmup', type=int, default=60000,
                        help='Steps before starting to optimize for morphology')
    parser.add_argument('--episodes-per-morpho', type=int, default=5,
                        help='Episodes to run of each morphology')
    parser.add_argument('--disc-warmup', type=int, default=20000,
                        help='Steps before starting to train SAC')
    parser.add_argument('--record-test', action="store_true",
                        help='Record tests (may be slow)')
    parser.add_argument('--load-warmup', action="store_true",
                        help='Load previously saved warmup data')
    parser.add_argument('--q-weight-decay', type=float, default=1e-5,
                        help='Q-function weight decay')
    parser.add_argument('--disc-weight-decay', type=float, default=1e-5,
                        help='Discriminator weight decay')
    parser.add_argument('--vae-scaler', type=float, default=1.,
                        help='Scaling term for VAE loss in SAIL')
    parser.add_argument('--pos-type', type=str, default="norm", choices=["abs", "rel", "norm", "skip"],
                        help="Which position marker coordinate to use (absolute, relative, normalized-relative, or skip to omit it)")
    parser.add_argument('--vel-type', type=str, default="rel", choices=["abs", "rel", "norm", "skip"],
                        help="Which velocity marker coordinate to use (absolute, relative, normalized-relative, or skip to omit it)")
    parser.add_argument('--expert-legs', type=int, nargs="+", default=[0, 1],
                        help="Which legs to use for marker matching on the demonstrator side")
    parser.add_argument('--policy-legs', type=int, nargs="+", default=[0, 1],
                        help="Which legs to use for marker matching on the imitator side")
    parser.add_argument('--expert-markers', type=int, nargs="+", default=[1, 2, 3],
                        help="Which markers to use for matching on the demonstrator side")
    parser.add_argument('--policy-markers', type=int, nargs="+", default=[1, 2, 3],
                        help="Which markers to use for matching on the imitator side")
    parser.add_argument('--learn-disc-transitions', action="store_true",
                        help="Learn discriminator using s, s' transitions")
    parser.add_argument('--train-distance-value', action="store_true",
                        help="Learn a separate distance value which is used to optimize morphology")
    parser.add_argument('--co-adapt', action="store_true",
                        help="Adapt morphology as well as behaviour")
    parser.add_argument('--expert-env-name', type=str, default=None,
                        help="Expert env name")
    parser.add_argument('--subject-id', type=int, default=8,
                        help="Expert subject name when using CMU dataset")
    parser.add_argument('--expert-episode-length', type=int, default=300,
                        help="Episode length for non-mocap expert data")
    parser.add_argument('--resume', type=str, default=None,
                        help="Resume from given policy")
    parser.add_argument('--torso-type', type=str, default=None, nargs="+",
                        help="Use torso velocity, position or skip")
    parser.add_argument('--head-type', type=str, default=None, nargs="+",
                        help="Use head velocity, position or skip")
    parser.add_argument('--head-wrt', type=str, default=None, nargs="+",
                    help="Use head with respect to body part (torso, butt)")
    parser.add_argument('--absorbing-state', action="store_true",
                        help="Replace terminal states with special absorbing states")
    parser.add_argument('--omit-done', action="store_true",
                        help="Simply set done=False always for learning purposes. Alternative to absorbing states.")
    parser.add_argument('--save-morphos', action="store_true",
                        help="Save morphology parameters and corresponding Wasserstein distances for later")
    parser.add_argument('--dist-optimizer', default=None, choices=["BO", "CMA", "RS"],
                        help="Co-adapt for Wasserstein distance, and optimize using algo.")
    parser.add_argument('--bo-gp-mean', choices=['Zero', 'Constant', 'Linear'], default='Zero')
    parser.add_argument('--acq-weight', type=float, default=2.,
                        help="BO LCB acquisition function exploration weight")
    parser.add_argument("--fixed-morpho", nargs="+", default=None, type=float)
    parser.add_argument('--normalize-obs', action="store_true",
                        help="Normalize observations for critic")
    parser.add_argument('--save-checkpoints', action="store_true",
                        help="Save buffer and models")

    args = parser.parse_args()

    wandb.init(name=args.algo, config=args)

    single_run(args)

