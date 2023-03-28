import itertools
import time
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ot
import pyswarms as ps
import torch
from sklearn.decomposition import PCA

matplotlib.use("agg")
import GPyOpt
import matplotlib.pyplot as plt
import numpy as np
import torch
from bo_mat.models.bo_gpymodel import GPDext
from bo_mat.models.build_gpy_model import get_kernel_module, get_mean_module
from sklearn.preprocessing import MinMaxScaler


def create_GPBO_model(args, X, Y):
    # Create a specific mean function - linear mean function
    mean_module = get_mean_module(args.mean, X, Y)

    # Create a specific kernel function - rbf + linear
    covar_module = get_kernel_module(args.kernel, X, args.add_bias, args.add_linear)

    # Create the GP-BO model
    model = GPDext(
        kernel=covar_module, mean_function=mean_module, optimizer=args.optimizer
    )

    return model


def get_new_candidates_BO(model, X, Y, env_name, min_task, max_task, acq_weight):
    # x_max, x_min = 2.*np.ones(3), 0.5*np.ones(3)#x_scaler.data_max_, x_scaler.data_min_

    # Domains
    # - Arm bandit
    #     space  =[{'name': 'var_1', 'type': 'bandit', 'domain': [(-1,1),(1,0),(0,1)]},
    #              {'name': 'var_2', 'type': 'bandit', 'domain': [(-1,4),(0,0),(1,2)]}]
    #
    #     - Continuous domain
    #     space =[ {'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality':1},
    #              {'name': 'var_2', 'type': 'continuous', 'domain':(-3,1), 'dimensionality':2},
    #              {'name': 'var_3', 'type': 'bandit', 'domain': [(-1,1),(1,0),(0,1)], 'dimensionality':2},
    #              {'name': 'var_4', 'type': 'bandit', 'domain': [(-1,4),(0,0),(1,2)]},
    #              {'name': 'var_5', 'type': 'discrete', 'domain': (0,1,2,3)}]
    #
    #     - Discrete domain
    #     space =[ {'name': 'var_3', 'type': 'discrete', 'domain': (0,1,2,3)}]
    #              {'name': 'var_3', 'type': 'discrete', 'domain': (-10,10)}]
    #
    #
    #     - Mixed domain
    #     space =[{'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality' :1},
    #             {'name': 'var_4', 'type': 'continuous', 'domain':(-3,1), 'dimensionality' :2},
    #             {'name': 'var_3', 'type': 'discrete', 'domain': (0,1,2,3)}]

    if env_name == "GaitTrackScaledHumanoid-v0":
        domain = [
            {"name": "var1", "type": "continuous", "domain": (0.5, 2.0)},
            {"name": "var_2", "type": "continuous", "domain": (0.5, 2.0)},
            {"name": "var_3", "type": "continuous", "domain": (0.5, 2.0)},
        ]
    elif env_name == "GaitTrackHalfCheetah-v0":
        domain = [
            {
                "name": f"var{i+1}",
                "type": "continuous",
                "domain": (min_task[i], max_task[i]),
            }
            for i in range(len(min_task))
        ]
    elif env_name == "GaitTrack2segHalfCheetah-v0":
        domain = [
            {
                "name": f"var{i+1}",
                "type": "continuous",
                "domain": (min_task[i], max_task[i]),
            }
            for i in range(len(min_task))
        ]
    else:
        raise NotImplementedError

    # X_init = np.array([[0.0], [0.5], [1.0]])
    # Y_init = func.f(X_init)

    # iter_count = 10
    # current_iter = 0
    # X_step = X_init
    # Y_step = Y_init

    # Maximization is not effective in this case so we need to change Y to negative values
    Y = Y

    bo_step = GPyOpt.methods.BayesianOptimization(
        f=None,
        domain=domain,
        X=X,
        Y=Y,
        model=model,
        normalize_Y=True,
        maximize=False,
        acquisition_type="LCB",
        acquisition_weight=acq_weight,
    )

    # Here is when the model is trained
    x_next = bo_step.suggest_next_locations()

    exploitation_lcb = GPyOpt.acquisitions.AcquisitionLCB(
        model=bo_step.model,
        space=bo_step.space,
        optimizer=bo_step.acquisition.optimizer,
        exploration_weight=0.0,
    )

    x_next_exploit = exploitation_lcb.optimize()[0]

    print(
        f"Proposed next X=:[{x_next}] with exploration {bo_step.acquisition.exploration_weight}"
    )

    return x_next, x_next_exploit, bo_step


def get_marker_info(
    info_dict,
    legs,
    marker_idx,
    *,
    pos_type="norm",
    vel_type="rel",
    torso_type=None,
    head_type=None,
    head_wrt=None,
):
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

    if "dones" in info_dict:
        eps = []
        i = 0
        for j, d in enumerate(info_dict["dones"]):
            if d:
                eps.append(all_data[i : j + 1])
                i = j + 1
        return eps, all_keys

    return all_data, all_keys


def visualize_morphos(morphos, optimized_or_not):
    X = np.stack(morphos)
    op = np.stack(optimized_or_not)

    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)

    fig = plt.figure(dpi=100, figsize=(8, 8))

    plt.scatter(
        X_transformed[op, 0], X_transformed[op, 1], c="green", label="Optimized"
    )
    plt.scatter(
        X_transformed[~op, 0], X_transformed[~op, 1], c="red", label="Exploration"
    )
    plt.legend()
    return fig


def obj(
    morpho_params,
    initial_states_torch,
    agent,
    evaluate_grads=False,
):
    if not type(morpho_params) == torch.Tensor:
        morpho_params = torch.as_tensor(
            morpho_params, device=initial_states_torch.device, dtype=torch.float32
        )

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
        print("Morpho grads")
        print(morpho_params.grad)
        return loss, morpho_params.grad.abs().sum().item()

    return loss


def obj_morpho_value(
    morpho_params,
    agent,
):
    if not type(morpho_params) == torch.Tensor:
        morpho_params = torch.as_tensor(
            morpho_params, device=agent.device, dtype=torch.float32
        )

    loss = agent.morpho_value(morpho_params).mean(-1)

    return loss


@torch.no_grad()
def plot_full_q_fn(fn, bounds, current_optima, real_lengths=None):
    optima = torch.as_tensor(current_optima).float()
    optima = optima.repeat(100, 1)

    fig = plt.figure(figsize=(12, 12), dpi=100)

    for i in range(len(bounds)):
        x = torch.linspace(bounds[i, 0], bounds[i, 1], 100).float()
        param_to_test = optima.clone()
        param_to_test[:, i] = x
        losses = fn(param_to_test.to("cuda:0"))
        assert losses.shape == (100,)
        plt.subplot(2, 3, i + 1)
        plt.axvline(current_optima[i], c="black", linestyle="--")
        if real_lengths and len(real_lengths) > i:
            plt.axvline(real_lengths[i], c="green", linestyle="--")
        plt.plot(x, losses)
        # plt.xlabel(f'Scale parameter {i+1}')
        # plt.ylabel('Negative value')

    return fig


def optimize_morpho_params_pso(
    agent, initial_states, bounds, use_distance_value=False, device="cpu"
):
    initial_states_torch = torch.as_tensor(
        np.stack(initial_states), dtype=torch.float32, device=device
    )
    # initial_states_torch = torch.as_tensor(replay.sample(1024)[0], dtype=torch.float32, device=device)

    # Subsample initial states
    n_samples = 1024
    # initial_states_torch = initial_states_torch[torch.randint(0, len(initial_states_torch), (n_samples, ))]
    initial_states_torch = initial_states_torch[-n_samples:]

    @torch.no_grad()
    def fn(x):
        if use_distance_value:
            losses = (
                obj_morpho_value(
                    x,
                    agent,
                )
                .cpu()
                .numpy()
            )
        else:
            losses = obj(x, None, initial_states_torch, agent).cpu().numpy()

        assert losses.shape == (x.shape[0],)
        torch.cuda.synchronize()
        return losses

    # bounds = (np.array([0.5]), np.array([2.]))
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}

    optimizer = ps.single.GlobalBestPSO(
        n_particles=250,
        dimensions=bounds.shape[0],
        options=options,
        bounds=(bounds[:, 0].numpy(), bounds[:, 1].numpy()),
        ftol=1e-7,
        ftol_iter=30,
    )
    cost, pos = optimizer.optimize(fn, iters=250)

    # print gradients. They should be zero-ish if we are at local optimum
    # _, grads_abs_sum = obj(pos, None, initial_states_torch, policy, q_function, evaluate_grads=True)

    fig = plot_full_q_fn(fn, bounds, pos)
    morpho_params = torch.tensor(pos)

    return cost, morpho_params, fig, 0


def _distance(training_transitions, demo):
    cost_train_distance = ot.dist(training_transitions, demo)
    distance = ot.emd2(
        [], [], cost_train_distance, numItermax=1000000, numThreads="max"
    )
    del cost_train_distance

    return distance


def compute_distance(training_transitions, demo, to_match):
    # Compute distances separately for velocities and positions
    pos_indices = []
    idx = 0
    for key in to_match:
        if "pos" in key:
            pos_indices += [idx, idx + 1, idx + 2]
        idx += 3

    pos_distance = _distance(training_transitions[:, pos_indices], demo[:, pos_indices])

    vel_indices = []
    idx = 0
    for key in to_match:
        if "vel" in key:
            vel_indices += [idx, idx + 1, idx + 2]
        idx += 3

    vel_distance = _distance(training_transitions[:, vel_indices], demo[:, vel_indices])

    return pos_distance, vel_distance


def handle_absorbing(
    feats,
    action,
    reward,
    next_feats,
    mask,
    marker_obs,
    next_marker_obs,
    obs_size,
    pwil_rewarder=None,
):
    marker_obs = np.concatenate([marker_obs, np.zeros(1)])
    if feats.shape[0] != obs_size:
        feats = np.concatenate([feats, np.zeros(1)])

    to_push = []

    if mask == 0.0:
        # next obs is absorbing
        next_marker_obs = np.zeros(marker_obs.shape[0])
        next_marker_obs[-1] = 1.0
        next_feats = np.zeros_like(feats)
        next_feats[-1] = 1.0

        # in addition add transition from absorbing to itself
        add_action = np.zeros_like(action)
        absorbing_state = next_marker_obs.copy()
        to_push.append(
            (
                next_feats,
                add_action,
                0.0,
                next_feats,
                1.0,
                1.0,
                absorbing_state,
                absorbing_state,
            )
        )
    else:
        next_feats = np.concatenate([next_feats, np.zeros(1)])
        next_marker_obs = np.concatenate([next_marker_obs, np.zeros(1)])

    if pwil_rewarder is not None:
        pwil_reward = pwil_rewarder.compute_reward({"observation": next_marker_obs})
        reward = pwil_reward

    to_push.append(
        (feats, action, reward, next_feats, 1.0, 1.0, marker_obs, next_marker_obs)
    )

    return to_push


def create_replay_data(env, marker_info_fn, agent, absorbing_state=True, steps=5000):
    to_push = []
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

            mask = 1.0

            if absorbing_state:
                to_push.extend(
                    handle_absorbing(
                        feats,
                        action,
                        reward,
                        next_feats,
                        mask,
                        marker_obs,
                        next_marker_obs,
                        agent.num_inputs + agent.num_morpho_obs,
                    )
                )
            else:
                to_push.append(
                    (
                        feats,
                        action,
                        reward,
                        next_feats,
                        mask,
                        marker_obs,
                        next_marker_obs,
                    )
                )

            state = next_state
            marker_obs = next_marker_obs

            step += 1
        print("Episode done", rsum)

    print(f"Took {time.time() - start_time:.2f}")
    return to_push


def bo_step(args, morphos, num_morpho, pos_train_distances, env):
    bo_args = SimpleNamespace()
    bo_args.mean = args.bo_gp_mean
    bo_args.kernel = "Matern52"
    bo_args.optimizer = "lbfgsb"
    bo_args.gp_type = "GPR"
    bo_args.acq_type = "LCB"
    bo_args.add_bias = 0
    bo_args.add_linear = 0

    # TODO change when resuming from pretrained
    prev_morphos_to_consider = 200
    if args.env_name == "GaitTrackHalfCheetah-v0":
        prev_morphos_to_consider = 200

    X = np.array(morphos).reshape(-1, args.episodes_per_morpho, num_morpho)[:, 0][
        -prev_morphos_to_consider:
    ]
    Y = (
        np.array(pos_train_distances)
        .reshape(-1, args.episodes_per_morpho)
        .mean(1, keepdims=True)[-prev_morphos_to_consider:]
    )

    model = create_GPBO_model(bo_args, X, Y)
    x_next, x_exploit_next, _ = get_new_candidates_BO(
        model,
        X,
        Y,
        args.env_name,
        env.min_task,
        env.max_task,
        args.acq_weight,
        None,
        acq_type="LCB",
    )
    morpho_params_np = x_next.flatten()
    optimized_morpho_params = x_exploit_next.flatten()
    print("Exploit morpho params", optimized_morpho_params)

    return morpho_params_np, optimized_morpho_params


def rs_step(args, num_morpho, morphos, pos_train_distances, min_task, max_task):
    # Average over same morphologies
    X = np.array(morphos).reshape(-1, args.episodes_per_morpho, num_morpho)[:, 0]
    Y = (
        np.array(pos_train_distances)
        .reshape(-1, args.episodes_per_morpho)
        .mean(1, keepdims=True)
    )

    curr = X[-1]
    if Y[-1] > Y[-2]:
        curr = X[-2]

    curr += np.random.normal(size=curr.shape) * 0.05
    curr = np.clip(curr, min_task, max_task)
    print("RS: new params", curr)

    # Second is always best found so far
    return curr, X[np.argmin(Y)]
