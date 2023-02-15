# Copyright 2022 (c) Anonymous authors - All Rights Reserved
#

import csv
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error


def evaluate_GPy_on_data(model, test_X, Y, writer, train_data=True, bayesian_opt=False):

    observed_pred = model.predict(test_X)

    ground_truth = Y
    if bayesian_opt:  # Correct for the bayesian optimisation, minimisation objective
        pred_Y_mean = -observed_pred[0]
    else:
        pred_Y_mean = observed_pred[0]
    pred_var = observed_pred[1]

    rmse_gp = mean_squared_error(ground_truth, pred_Y_mean, multioutput='raw_values', squared=False)
    sum_result = np.sum(0.5 * np.log(2 * np.pi * pred_var) + (ground_truth - pred_Y_mean) ** 2 / (2 * pred_var), axis=1)
    nll = np.mean(sum_result, axis=0)  # Sum over the timesteps

    if train_data:
        print(F"Train RMSE={rmse_gp[0]}")
        print(F"Train nll={nll}")
        writer.add_scalar('train_RMSE', rmse_gp[0], global_step=0)
        writer.add_scalar('train_NLL', nll, global_step=0)
    else:
        writer.add_scalar('test_RMSE', rmse_gp[0], global_step=0)
        writer.add_scalar('test_NLL', nll, global_step=0)
