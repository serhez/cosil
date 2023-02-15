# Copyright 2022 (c) Anonymous authors - All Rights Reserved
#
import numpy as np
import GPy


def get_mean_module(mean_ft, X, Y):

    if mean_ft == 'Zero':
        mean_module = None
    elif mean_ft == 'Constant':
        mean_module = GPy.mappings.Constant(X.shape[1], Y.shape[1])
    elif mean_ft == 'Linear':
        mean_module = GPy.mappings.Linear(X.shape[1], Y.shape[1])
    else:
        raise NotImplementedError(f'Mean function not supported: {mean_ft}, use Zero/Constant/Linear')

    return mean_module


def get_kernel_module(kernel_ft, X, add_bias: bool, add_linear: bool):

    if kernel_ft == 'Exponential':
        covar_module = GPy.kern.Exponential(X.shape[1], ARD=True)
    elif kernel_ft == 'Matern32':
        covar_module = GPy.kern.Matern32(X.shape[1], ARD=True)
    elif kernel_ft == 'Matern52':
        covar_module = GPy.kern.Matern52(X.shape[1], ARD=True)
    elif kernel_ft == 'RBF':
        covar_module = GPy.kern.RBF(X.shape[1], ARD=True)
    elif kernel_ft == 'ExpQuad':
        covar_module = GPy.kern.ExpQuad(X.shape[1], ARD=True)
    elif kernel_ft == 'OU':
        covar_module = GPy.kern.OU(X.shape[1], ARD=True)
    else:
        raise NotImplementedError(f'Kernel not supported: {kernel_ft},'
                                  f' use Exponential/Matern32/Matern52/RBF/ExpQuad/OU')

    # We add to the covariance module a linear kernel to be able to get the noise
    # as GPyTorch kernels have no noise by default
    if add_bias:
        covar_module = covar_module + GPy.kern.Bias(X.shape[1])
    if add_linear:
        covar_module = covar_module + GPy.kern.Linear(X.shape[1])

    return covar_module


def build_gp_model(X, Y, mean_ft, kernel_ft, gp_type, add_bias, add_linear):

    # Create a specific mean function - linear mean function
    mean_module = get_mean_module(mean_ft, X, Y)

    # Create a specific kernel function - rbf + linear
    covar_module = get_kernel_module(kernel_ft, X, add_bias, add_linear)

    # Create the GP model
    if gp_type == 'GPR':
        model = GPy.models.GPRegression(X, Y, kernel=covar_module, mean_function=mean_module)
    elif gp_type == 'WarpedGP':
        warp_f = GPy.util.warping_functions.LogFunction()
        model = GPy.models.WarpedGP(X, Y, kernel=covar_module, warping_function=warp_f)
    else:
        raise NotImplementedError(f'GP type not supported: {gp_type}, use GPR/WarpedGP')

    return model
