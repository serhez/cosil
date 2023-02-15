# Copyright 2022 (c) Anonymous authors - All Rights Reserved
#

import numpy as np
import GPy
from GPyOpt.models import BOModel


class GPDext(BOModel):
    """
    General template to create a new GPyOPt surrogate model

    :param normalize Y: wheter the outputs are normalized (default, false)

    """

    # SET THIS LINE TO True of False DEPENDING IN THE ANALYTICAL GRADIENTS OF THE PREDICTIONS ARE AVAILABLE OR NOT
    analytical_gradient_prediction = True

    def __init__(self, kernel, mean_function, optimizer='lbfgsb', exact_feval=False,
                 optimize_restarts=1, max_iters=1000, verbose=True):

        # ---
        # ADD TO self... THE REST OF THE PARAMETERS OF YOUR MODEL
        # ---
        self.kernel = kernel
        self.mean_function = mean_function
        self.optimizer = optimizer
        self.verbose = verbose
        self.model = None
        self.exact_feval = exact_feval
        self.max_iters = max_iters
        self.optimize_restarts = optimize_restarts

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """
        self.X = X
        self.Y = Y

        self.model = GPy.models.GPRegression(self.X, self.Y,
                                             kernel=self.kernel, mean_function=self.mean_function)

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """
        Updates the model with new observations.
        """
        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            self.model.set_XY(X_all, Y_all)

        # --- update the model maximizing the marginal likelihood.
        if self.optimize_restarts == 1:
            self.model.optimize(optimizer=self.optimizer, messages=self.verbose)
        else:
            self.model.optimize_restarts(num_restarts=self.optimize_restarts, optimizer=self.optimizer,
                                         max_iters=self.max_iters, verbose=self.verbose)

    def _predict(self, X, full_cov):
        """
        Preditions with the model. Returns posterior means m and standard deviations s at X.
        """

        if X.ndim == 1:
            X = X[None, :]
        m, v = self.model.predict(X, full_cov=full_cov)
        v = np.clip(v, 1e-10, np.inf)
        return m, v

    def predict(self, X):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        m, v = self._predict(X, False)
        # We can take the square root because v is just a diagonal matrix of variances
        return m, np.sqrt(v)

    def predict_covariance(self, X):
        """
        Predicts the covariance matric for points in X.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        _, v = self._predict(X, True)
        return v

    def get_fmin(self):
        return self.model.predict(self.model.X)[0].min()

    def predict_withGradients(self, X):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
        """
        if X.ndim==1: X = X[None,:]
        m, v = self.model.predict(X)
        v = np.clip(v, 1e-10, np.inf)
        dmdx, dvdx = self.model.predictive_gradients(X)
        dmdx = dmdx[:, :, 0]
        dsdx = dvdx / (2*np.sqrt(v))

        return m, np.sqrt(v), dmdx, dsdx

    def copy(self):
        """
        Makes a safe copy of the model.
        """
        copied_model = GPDext(kernel=self.model.kern.copy(),
                              mean_function=self.mean_function.copy(),
                              exact_feval=self.exact_feval,
                              optimizer=self.optimizer,
                              max_iters=self.max_iters,
                              optimize_restarts=self.optimize_restarts,
                              verbose=self.verbose)

        copied_model._create_model(self.model.X, self.model.Y)
        copied_model.updateModel(self.model.X, self.model.Y, None, None)
        return copied_model

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        return np.atleast_2d(self.model[:])

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        return self.model.parameter_names_flat().tolist()

    def get_covariance_between_points(self, x1, x2):
        """
        Given the current posterior, computes the covariance between two sets of points.
        """
        return self.model.posterior_covariance_between_points(x1, x2)


