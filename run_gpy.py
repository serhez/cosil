import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import argparse
import sys
import math

import wandb
import GPy
import GPyOpt
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from bo_mat.models.build_gpy_model import get_mean_module, get_kernel_module
from bo_mat.models.bo_gpymodel import GPDext
from bo_mat.utils.data_utils import evaluate_GPy_on_data
import GPy.plotting.gpy_plot.gp_plots

def load_dataset(dataset_dir, double_opt, plot_dataset, use_log: bool, scale_input: bool):
    """Load the Matlab files in the specified directory"""

    mat_files = os.listdir(dataset_dir)
    X, Y = [], []
    for file in mat_files:
        filename = dataset_dir+'/'+file
        if file[-4:] != '.mat':
            continue
        mat = loadmat(filename)  # Load the MATLAB test data

        # Extract the velociy of the test, settling time, stiffness and max dist
        vec_dist, vec_force, f_r, vel, st, stiffness, elongation, toughness = extract_dextran_info(mat, filename)
        X.append(np.array([[vel, f_r, st]]))

        if double_opt:
            Y.append(np.array([[toughness, elongation]]))
        else:
            Y.append(np.array([toughness]))
        if plot_dataset:
            plot_dextran(vec_dist, vec_force, f_r, vel, st, stiffness, elongation, toughness)

    X = np.vstack(X)
    Y = np.vstack(Y)

    x_scaler, _ = get_dextran_scaler(X, Y)

    if not scale_input:
        X[:, 0] = X[:, 0] / 10  # If we divide the velocity by 10 the performance highly increases
        x_scaler.data_max_[0] = x_scaler.data_max_[0]/10
        x_scaler.data_min_[0] = x_scaler.data_min_[0] / 10
        x_scaler.data_range_[0] = x_scaler.data_range_[0]/10

    if use_log:
        Y = np.log(Y)

    return X, Y, x_scaler


def test_GP_model(model, zval, x_scaler: MinMaxScaler, scale_input: bool):

    # Create test data in the range of velocities and settling time used
    #x_max, x_min = x_scaler.data_max_, x_scaler.data_min_
    vel_max, fr_max, st_max = 2, 2, 2
    vel_min, fr_min, st_min = 0.5,0.5,0.5

    # The velocity range is 50-500 so 10 samples is enough
    test_vel = np.linspace(vel_min, vel_max, 100)
    test_fr = np.linspace(fr_min, fr_max, 100)
    
    xs, ys = np.meshgrid(test_vel, test_fr)
    zs = zval * np.ones_like(xs)
    test_data = np.stack((xs.flatten(), ys.flatten(), zs.flatten())).T

    if scale_input:
        test_data = x_scaler.transform(test_data)

    observed_pred = model.predict(test_data)

    return observed_pred, test_data


def create_GPBO_model(args, X, Y):
    # Create a specific mean function - linear mean function
    mean_module = get_mean_module(args.mean, X, Y)

    # Create a specific kernel function - rbf + linear
    covar_module = get_kernel_module(args.kernel, X, args.add_bias, args.add_linear)

    # Create the GP-BO model
    model = GPDext(kernel=covar_module, mean_function=mean_module, optimizer=args.optimizer)

    return model


def get_new_candidates_BO(model, X, Y, env_name, min_task, max_task, acq_weight, x_scaler: MinMaxScaler, acq_type):

    x_max, x_min = 2.*np.ones(3), 0.5*np.ones(3)#x_scaler.data_max_, x_scaler.data_min_

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
        domain = [{'name': 'var1', 'type': 'continuous', 'domain': (0.5, 2.)},
                {'name': 'var_2', 'type': 'continuous', 'domain': (0.5, 2.)},
                {'name': 'var_3', 'type': 'continuous', 'domain': (0.5, 2.)}]
    elif env_name == "GaitTrackHalfCheetah-v0":
        domain = [{'name': f'var{i+1}', 'type': 'continuous', 'domain': (min_task[i], max_task[i])} for i in range(len(min_task))]
    elif env_name == "GaitTrack2segHalfCheetah-v0":
        domain = [{'name': f'var{i+1}', 'type': 'continuous', 'domain': (min_task[i], max_task[i])} for i in range(len(min_task))]
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

    bo_step = GPyOpt.methods.BayesianOptimization(f=None, domain=domain, X=X, Y=Y, model=model,
                                                  normalize_Y=True, maximize=False, acquisition_type="LCB", acquisition_weight=acq_weight)

    # Here is when the model is trained
    x_next = bo_step.suggest_next_locations()

    exploitation_lcb = GPyOpt.acquisitions.AcquisitionLCB(model=bo_step.model, space=bo_step.space,
                                                        optimizer=bo_step.acquisition.optimizer,
                                                        exploration_weight=0.0)

    x_next_exploit = exploitation_lcb.optimize()[0]
    
    print(f"Proposed next X=:[{x_next}] with exploration {bo_step.acquisition.exploration_weight}")
    
    return x_next, x_next_exploit, bo_step


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='/home/mulerod1/projects/bo_mat/dataset/dextran_dataset')
    parser.add_argument('--plot', action='store_true', default=False,
                        help="Whether to plot the material force VS deformation")
    parser.add_argument('--plot_dataset', action='store_true', default=False,
                        help="Whether to plot the material force VS deformation")
    parser.add_argument('--double_opt', action='store_true', default=False,
                        help='If enabled BO will optimise both max deformation and stiffness, '
                             'otherwise it will optimise only max deformation.')
    parser.add_argument('--mean', type=str, choices=['Zero', 'Constant', 'Linear'], default='Constant')
    parser.add_argument('--kernel', type=str, choices=['Exponential', 'Matern32', 'Matern52', 'RBF', 'ExpQuad', 'OU'],
                        default='Matern52')
    parser.add_argument('--optimizer', type=str, choices=['scg', 'lbfgs', 'sgd', 'lbfgsb'], default='lbfgsb',
                        help='Available optimizers are scaled conjugate gradient, L-BFGS, L-BFGS-B and SGD')

    parser.add_argument('--gp_type', type=str, choices=['GPR', 'WarpedGP'], default='GPR')
    parser.add_argument('--add_bias', type=int, choices=[0, 1], default=0,
                        help='Add a Linear kernel to the base kernel')
    parser.add_argument('--add_linear', type=int, choices=[0, 1], default=0,
                        help='Add a Linear kernel to the base kernel')
    parser.add_argument('--scale_input', type=int, choices=[0, 1], default=0,
                        help='Scale X to the Min-Max range.')
    parser.add_argument('--acq_type', type=str, choices=['EI', 'MPI', 'LCB'], default='LCB',
                        help='Choose the acquisition function, EI/MPI/LCB'
                             ' Expected Improvement/Maximum Probability of Improvement/Lower Confidence Bound')
    # parser.add_argument('--maxiter', type=int, default=1000)  # By default GPy uses 10000
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # Get arguments
    args = get_parser()
    args.scale_input = bool(args.scale_input)
    args.add_bias = bool(args.add_bias)
    args.add_linear = bool(args.add_linear)

    # Get the training dataset
    if args.gp_type == 'WarpedGP':  # If we use the WarpedGP we do not need to convert the targets
        use_log = False
    else:
        use_log = True

    #X, Y, x_scaler = load_dataset(args.train_dir, args.double_opt, args.plot_dataset,
    #                              use_log=use_log, scale_input=args.scale_input)
    morphos = torch.load('morphos.pt')
    X = torch.tensor(morphos["morphos"]).float().numpy()
    Y = torch.tensor(morphos["distances"]).view(-1, 1).float().numpy()   

    x_scaler = MinMaxScaler((0.5, 2.))

    model_name = f"{args.mean}_{args.kernel}_{args.gp_type}_b{args.add_bias}_l{args.add_linear}_acq_{args.acq_type}"
    model_name = model_name.replace('.', '_')

    plt.plot(X[:, 2], label='2')
    plt.plot(X[:, 0], label='0')
    plt.plot(X[:, 1], label='1')
    plt.legend()
    plt.savefig('lol.jpg')
    exit()
    # wandb.tensorboard.patch(root_logdir=model_name)
    # wandb.init(project="BO_mat", entity='dblanm', config={'log_name': model_name})
    # wandb.config.update(args)
    #writer = SummaryWriter(log_dir=model_name)

    # Plot the unnormalised data
    if args.plot:
        plot_dextran_dataset(X, Y)

    # If we normalise the input replace X
    if args.scale_input:
        X = x_scaler.transform(X)

    model = create_GPBO_model(args, X, Y)
    x_next, x_exploit_next, bo_step, exploit_lcb = get_new_candidates_BO(model, X, Y, "GaitTrackScaledHumanoid-v0",None, None, None, acq_type="LCB")

    optimised_model = bo_step.model

    # Get the RMSE & NLL for the GP model on the training data
    evaluate_GPy_on_data(optimised_model, X, Y, None, bayesian_opt=True)


    plt.figure(figsize=(15,5), dpi=150)
    for i, zval in enumerate([0.5, 1., 1.5, 2.]):
        plt.subplot(1,4,i+1)
        preds, test_X = test_GP_model(optimised_model, zval, None, False)

        #ucb = (-preds[0] + 5*preds[1]).reshape(100, 100)
        ucb = exploit_lcb.acquisition_function(test_X).reshape(100, 100)
        print(exploit_lcb.exploration_weight)        
        #ucb = preds[0].reshape(100, 100)
        
        img = plt.contour(test_X[:, 0].reshape(100, 100), test_X[:, 1].reshape(100, 100), ucb)
        plt.colorbar(img)
    
    #optimised_model.model.plot(fixed_inputs=((2, 0.5), ), legend=True, resolution=200)
    #ax = plt.gca() #get current axes
    #mappable = ax.collections[0] #this is specific for what the surface call returns
    #plt.colorbar(mappable)
    
    plt.savefig(f'plot.jpg')
    exit()
    
    # Test the GP with the whole range Min-Max of each of the material properties
    pred_Y, test_X = test_GP_model(optimised_model, x_scaler, args.scale_input)

    # Get the acquisition function values on the test set X
    acq_test_X = bo_step.acquisition.acquisition_function(test_X)

    # if not args.double_opt:  # We only plot the toughness
    if args.scale_input:
        plot_dextran_toughness_gpy(x_scaler.inverse_transform(X), Y, x_scaler.inverse_transform(test_X), pred_Y,
                                   title=model_name + '/' + model_name, writer=writer)
    else:
        # plot_dextran_toughness_gpy(X, Y, test_X, pred_Y,
        #                            title=model_name+'/'+model_name, writer=writer)
        plot_dextran_acq_gpy(X, Y, test_X, pred_Y, acq_test_X, x_next, acq_type=args.acq_type,
                             title=model_name+'/'+model_name, writer=writer)

    a = 0












