from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import norm
from scipy.optimize import minimize
import sys
import pandas as pd
import numpy as np
import sys

from matplotlib import pyplot as plt
from matplotlib.patches import Patch


class GridSearch():

    def __init__(self, objective, x_start, n_iter):
        self.objective = objective
        self.n_iter = n_iter
        self.best_samples_ = pd.DataFrame(columns = ['y'])
        self.x_start = x_start

    def optimize(self):
        y_max = -2.0
        x_next = self.x_start
        for i in range(self.n_iter):
            y_next = self.objective(x_next)
            if y_next > y_max:
                y_max = y_next
            x_next[0, 0] = x_next[0, 0] + 1
            if x_next[0, 0] > 27:
                x_next[0, 0] = 0
                x_next[0, 1] = x_next[0, 1] + 1
            self.best_samples_ = self.best_samples_.append({"y": y_max}, ignore_index=True)


class NaiveBayesianOptimizer():
    """Bayesian optimizer.

    Args:
        objective: callable objective function
        constraint_f: callable constraint function
        constraint: value which the constraint function must not fall below
        variables: dictionary of optimizations variables
        n_init: number of initial samples
        n_iter: number of iterations
    """

    def __init__(self, objective, constraint_f, constraint, variables, n_init, n_iter, acquisition_f, batch_size=32):
        self.objective = objective
        self.constraint_f = constraint_f
        self.constraint = constraint
        self.variables = variables
        self.n_init = n_init
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.acquisition_f = acquisition_f
        self.gauss_pr = GaussianProcessRegressor(kernel=RBF(length_scale_bounds=(1e-37, 1e37)), n_restarts_optimizer=self.batch_size, normalize_y=True)
        self.best_samples_ = pd.DataFrame(columns = ['x_0', 'x_1', 'x_2', 'x_3', 'y', 'ei'])
        self.distances_ = []

    def _get_expected_improvement(self, x_new):
        # First get estimate from Gaussian surrogate
        mean_y_new, sigma_y_new = self.gauss_pr.predict(np.array([x_new]), return_std=True)
        sigma_y_new = sigma_y_new.reshape(-1,1)
        if sigma_y_new == 0.0:
            return 0.0
        
        mean_y = self.gauss_pr.predict(self.x_init)
        max_mean_y = np.max(mean_y)
        z = (mean_y_new - max_mean_y) / sigma_y_new
        exp_imp = (mean_y_new - max_mean_y) * norm.cdf(z) + sigma_y_new * norm.pdf(z)
        return exp_imp

    def _get_probability_improvement(self, x_new):
        mean_y_new, sigma_y_new = self.gauss_pr.predict(np.array([x_new]), return_std=True)
        sigma_y_new = sigma_y_new.reshape(-1, 1)
        if sigma_y_new == 0.0:
            return 0.0
        
        mean_y = self.gauss_pr.predict(self.x_init)
        max_mean_y = np.max(mean_y)
        pi = (mean_y_new - max_mean_y) / sigma_y_new

        return pi

    def _get_upper_confidence_bound(self, x_new):
        mean_y_new, sigma_y_new = self.gauss_pr.predict(np.array([x_new]), return_std=True)
        sigma_y_new = sigma_y_new.reshape(-1, 1)
        if sigma_y_new == 0.0:
            return 0.0

        return mean_y_new + sigma_y_new

    def _acquisition_function(self, x):
        if self.acquisition_f == 'ei':
            return -self._get_expected_improvement(x).ravel()
        elif self.acquisition_f == 'pi':
            return -self._get_probability_improvement(x).ravel()
        elif self.acquisition_f == 'ucb':
            return -self._get_upper_confidence_bound(x).ravel()


    def _initialize(self):
        self.x_init = np.random.random((self.n_init, len(self.variables)))
        for idx, var in enumerate(self.variables):
            self.x_init[:, idx] = np.round(self.x_init[:, idx] * np.abs(self.variables[var][0] - self.variables[var][1]) + self.variables[var][0])
        self.y_init = self.objective(self.x_init)
        self.c_init = self.constraint_f(self.x_init)


    def _get_next_probable_point(self):
        min_ei = float(sys.maxsize)
        x_optimal = None
        scale = 1
        for x_start in (np.random.random((self.batch_size, self.x_init.shape[1])) * scale):
            response = minimize(fun=self._acquisition_function, x0=x_start, method='CG')
            if response.fun < min_ei:
                min_ei = response.fun
                x_optimal = response.x
        return x_optimal, min_ei

    def _extend_prior_with_posterior_data(self, x, y, c):
        self.x_init = np.append(self.x_init, x, axis=0)
        self.y_init = np.append(self.y_init, np.array(y), axis=0)
        self.c_init = np.append(self.c_init, np.array(c), axis=0)


    def optimize(self):
        self._initialize()
        y_max_ind = np.argmax(self.y_init)
        y_max = -2.0
        optimal_x = self.x_init[y_max_ind]
        optimal_ei = None
        for i in range(self.n_iter):
            scaler = StandardScaler()
            scaler = MinMaxScaler((-1, 1))
            scaler.fit(self.x_init)
            self.gauss_pr.fit(scaler.transform(self.x_init), self.y_init ** 2)
            x_next, ei = self._get_next_probable_point()
            x_next = np.clip(x_next, -1, 1)
            x_next = np.round(scaler.inverse_transform(x_next.reshape(1, -1)))
            y_next = self.objective(x_next)
            c_next = self.constraint_f(x_next)
            self._extend_prior_with_posterior_data(x_next, y_next, c_next)


            if y_next > y_max and c_next <= self.constraint:
                y_max = y_next
                optimal_x = x_next
                optimal_ei = ei

            if i == 0:
                prev_x = x_next

            else:
                self.distances_.append(np.linalg.norm(prev_x - x_next))
                prev_x = x_next
            self.best_samples_ = self.best_samples_.append({"y": y_max, "ei": optimal_ei}, ignore_index=True)

        return optimal_x, y_max

   

class BayesianOptimizer():
    """Bayesian optimizer.

    Args:
        objective: callable objective function
        constraint_f: callable constraint function
        constraint: value which the constraint function must not fall below
        variables: dictionary of optimizations variables
        n_init: number of initial samples
        n_iter: number of iterations
    """

    def __init__(self, objective, constraint_f, constraint, variables, n_init, n_iter, acquisition_f, batch_size=32):
        self.objective = objective
        self.constraint_f = constraint_f
        self.constraint = constraint
        self.variables = variables
        self.n_init = n_init
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.acquisition_f = acquisition_f
        self.gauss_pr = GaussianProcessRegressor(kernel=RBF(length_scale_bounds=(1e-37, 1e37)), n_restarts_optimizer=self.batch_size, normalize_y=True)
        self.constraint_pr = GaussianProcessRegressor(kernel=RBF(length_scale_bounds=(1e-37, 1e37)), n_restarts_optimizer=self.batch_size)
        self.best_samples_ = pd.DataFrame(columns = ['x_0', 'x_1', 'x_2', 'x_3', 'y', 'ei'])
        self.distances_ = []

    def _get_expected_improvement(self, x_new):
        # First get estimate from Gaussian surrogate
        mean_y_new, sigma_y_new = self.gauss_pr.predict(np.array([x_new]), return_std=True)
        sigma_y_new = sigma_y_new.reshape(-1,1)
        #print(mean_y_new)
        #print(sigma_y_new)
        if sigma_y_new == 0.0:
            return 0.0
        mean_c_new, sigma_c_new = self.constraint_pr.predict(np.array([x_new]), return_std=True)
        pf = norm.cdf(self.constraint, loc=mean_c_new, scale=sigma_c_new)
        
        
        mean_y = self.gauss_pr.predict(self.x_init)
        max_mean_y = np.max(mean_y)
        z = (mean_y_new - max_mean_y) / sigma_y_new
        exp_imp = (mean_y_new - max_mean_y) * norm.cdf(z) + sigma_y_new * norm.pdf(z)
        return exp_imp
        #return pf * exp_imp

    def _get_probability_improvement(self, x_new):
        mean_y_new, sigma_y_new = self.gauss_pr.predict(np.array([x_new]), return_std=True)
        sigma_y_new = sigma_y_new.reshape(-1, 1)
        if sigma_y_new == 0.0:
            return 0.0
        
        mean_y = self.gauss_pr.predict(self.x_init)
        max_mean_y = np.max(mean_y)
        pi = (mean_y_new - max_mean_y) / sigma_y_new


        mean_c_new, sigma_c_new = self.constraint_pr.predict(np.array([x_new]), return_std=True)
        pf = norm.cdf(self.constraint, loc=mean_c_new, scale=sigma_c_new)
        return pi
        #return pf * pi

    def _get_upper_confidence_bound(self, x_new):
        mean_y_new, sigma_y_new = self.gauss_pr.predict(np.array([x_new]), return_std=True)
        sigma_y_new = sigma_y_new.reshape(-1, 1)
        if sigma_y_new == 0.0:
            return 0.0

        mean_c_new, sigma_c_new = self.constraint_pr.predict(np.array([x_new]), return_std=True)
        pf = norm.cdf(self.constraint, loc=mean_c_new, scale=sigma_c_new)
        return mean_y_new + sigma_y_new
        #return pf * (mean_y_new + sigma_y_new)  


    def _acquisition_function(self, x):
        if self.acquisition_f == 'ei':
            return -self._get_expected_improvement(x).ravel()
        elif self.acquisition_f == 'pi':
            return -self._get_probability_improvement(x).ravel()
        elif self.acquisition_f == 'ucb':
            return -self._get_upper_confidence_bound(x).ravel()


    def _initialize(self):
        self.x_init = np.random.random((self.n_init, len(self.variables)))
        for idx, var in enumerate(self.variables):
            self.x_init[:, idx] = np.round(self.x_init[:, idx] * np.abs(self.variables[var][0] - self.variables[var][1]) + self.variables[var][0])
        self.y_init = self.objective(self.x_init)
        self.c_init = self.constraint_f(self.x_init)


    def _get_next_probable_point(self):
        min_ei = float(sys.maxsize)
        x_optimal = None
        scale = 1
        for x_start in (np.random.random((self.batch_size, self.x_init.shape[1])) * scale):
            #response = minimize(fun=self._acquisition_function, x0 =x_start, method='L-BFGS-B')
            response = minimize(fun=self._acquisition_function, x0=x_start, method='CG')
            #print(response)
            #if response.fun[0] < min_ei:
            if response.fun < min_ei:
                #min_ei = response.fun[0]
                min_ei = response.fun
                x_optimal = response.x
                #print(response)
        #print(f'X_OPT: {x_optimal}')
        return x_optimal, min_ei

    def _extend_prior_with_posterior_data(self, x, y, c):
        self.x_init = np.append(self.x_init, x, axis=0)
        self.y_init = np.append(self.y_init, np.array(y), axis=0)
        self.c_init = np.append(self.c_init, np.array(c), axis=0)


    def optimize(self):
        self._initialize()
        y_max_ind = np.argmax(self.y_init)
        #y_max = self.y_init[y_max_ind]
        y_max = -2.0
        optimal_x = self.x_init[y_max_ind]
        optimal_ei = None
        for i in range(self.n_iter):
            scaler = StandardScaler()
            scaler = MinMaxScaler((-1, 1))
            scaler.fit(self.x_init)
            self.gauss_pr.fit(scaler.transform(self.x_init), self.y_init ** 2)
            self.constraint_pr.fit(scaler.transform(self.x_init), self.c_init <= self.constraint)
            x_next, ei = self._get_next_probable_point()
            x_next = np.clip(x_next, -1, 1)
            #for j in range(0,100,20):
            #    for k in range(0,100,20):
            #        x_next[0] = j
            #        x_next[2] = k
            #        y_next = self.target_func(np.array([x_next]))
            #        c_next = self.c_func(np.array([x_next]))
            #        self._extend_prior_with_posterior_data(x_next,y_next,c_next)
            x_next = np.round(scaler.inverse_transform(x_next.reshape(1, -1)))
            y_next = self.objective(x_next)
            c_next = self.constraint_f(x_next)
            self._extend_prior_with_posterior_data(x_next, y_next, c_next)


            if y_next > y_max and c_next <= self.constraint:
                y_max = y_next
                optimal_x = x_next
                optimal_ei = ei

            if i == 0:
                prev_x = x_next

            else:
                self.distances_.append(np.linalg.norm(prev_x - x_next))
                prev_x = x_next
            self.best_samples_ = self.best_samples_.append({"y": y_max, "ei": optimal_ei}, ignore_index=True)

        return optimal_x, y_max

TD = '0.333333'
TD = sys.argv[1]
RISK_OOD_FIRST = np.load(f'risk_ood_first_{TD}.npy')
U_OOD_FIRST = np.load(f'U_ood_first_{TD}.npy')

#RISK_YOLO_FIRST = np.load(f'risk_{TD}.npy')
#U_YOLO_FIRST = np.load(f'U_{TD}.npy')
#RISK_SKIP_OOD_FIRST = np.load(f'risk_{TD}_skip.npy')
#U_SKIP_OOD_FIRST = np.load(f'U_{TD}_skip.npy')
#RISK_SKIP_YOLO_FIRST = np.load(f'risk_{TD}_skip_yolo_first.npy')
#U_SKIP_YOLO_FIRST = np.load(f'U_{TD}_skip_yolo_first.npy')



def target_old(x):
    out = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        ood_tau = min(int(x[i,0]), 99)
        ood_size = min(int(x[i,1]), 27)
        yolo_tau = min(int(x[i,2]), 99)
        yolo_size = min(int(x[i,3]), 13)
        out[i] = -RISK_OOD_FIRST[ood_tau, ood_size, yolo_tau, yolo_size]
    return out

def test_target(x):
    out = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        out[i] = - x[i, 0] ** 2 - x[i, 1] ** 2
    return out

def target_new(x):
    out = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        ood_size = min(int(x[i, 0]), 27)
        yolo_size = min(int(x[i, 1]), 13)
        out[i] = -np.amin(RISK_OOD_FIRST[:, ood_size, :, yolo_size])
    return out

def get_utilization(x):
    #if len(x.shape) == 1:
    #    return U[int(x[0]), int(x[1]), int(x[2]), int(x[3])]
    out = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        ood_size = min(int(x[i, 0]), 27)
        yolo_size = min(int(x[i, 1]), 13)
        out[i] = np.amin(U_OOD_FIRST[:, ood_size, :, yolo_size])
    return out     


N_TRIES = 100
N_ITER = 150
bayes_opt_ucb = np.zeros((N_TRIES, N_ITER))
bayes_opt_pi = np.zeros((N_TRIES, N_ITER))
bayes_opt_ei = np.zeros((N_TRIES, N_ITER))
naive_bayes = np.zeros((N_TRIES, N_ITER))
grid_search = np.zeros((N_TRIES, N_ITER))
#grid_search = GridSearch(objective=target_new, n_iter=N_ITER)
#grid_search.optimize()
for i in range(N_TRIES):
    """
    naive_opt = NaiveBayesianOptimizer(
        objective = target_old,
        constraint_f = get_utilization,
        constraint = 0.69,
        variables={'a': (0, 99), 'b': (0,27), 'c': (0,99), 'd': (0,13)},
        n_init=5,
        n_iter=N_ITER,
        acquisition_f='ei',
        batch_size=10
    )
    naive_opt.optimize()
    naive_bayes[i, :] = naive_opt.best_samples_['y'].to_numpy()
    grid_opt = GridSearch(objective=target_new, x_start=(13 * np.random.rand(1,2)).astype(np.uint8), n_iter=N_ITER)
    grid_opt.optimize()
    grid_search[i, :] = grid_opt.best_samples_['y'].to_numpy()
    """
    optimizer_ucb = BayesianOptimizer(
        objective = target_new,
        constraint_f = get_utilization,
        constraint = 0.999,
        variables={'a': (0, 13), 'b': (6, 13)},
        n_init=5,
        n_iter=N_ITER,
        acquisition_f='ei',
        batch_size=10
    )
    optimizer_ucb.optimize()
    optimizer_ucb2 = BayesianOptimizer(
        objective = target_new,
        constraint_f = get_utilization,
        constraint = 0.999,
        variables={'a': (0, 13), 'b': (0, 6)},
        n_init=5,
        n_iter=N_ITER,
        acquisition_f='ei',
        batch_size=10
    )
    optimizer_ucb2.optimize()
    optimizer_ucb3 = BayesianOptimizer(
        objective = target_new,
        constraint_f = get_utilization,
        constraint = 0.999,
        variables={'a': (13, 27), 'b': (0, 6)},
        n_init=5,
        n_iter=N_ITER,
        acquisition_f='ei',
        batch_size=10
    )
    optimizer_ucb3.optimize()
    optimizer_ucb4 = BayesianOptimizer(
        objective = target_new,
        constraint_f = get_utilization,
        constraint = 0.999,
        variables={'a': (13, 27), 'b': (6, 13)},
        n_init=5,
        n_iter=N_ITER,
        acquisition_f='ei',
        batch_size=10
    )
    optimizer_ucb4.optimize()
    bayes_opt_ucb[i, :] = np.maximum.reduce([optimizer_ucb.best_samples_['y'].to_numpy(), optimizer_ucb2.best_samples_['y'].to_numpy(), optimizer_ucb3.best_samples_['y'].to_numpy(), optimizer_ucb4.best_samples_['y'].to_numpy()])
    """
    naive_opt = BayesianOptimizer(
        objective = target_new,
        constraint_f = get_utilization,
        constraint = 0.999,
        variables={'a': (0, 27), 'b': (0, 13)},
        n_init=5,
        n_iter=N_ITER,
        acquisition_f='pi',
        batch_size=10
    )
    optimizer_pi.optimize()
    bayes_opt_pi[i, :] = optimizer_pi.best_samples_['y'].to_numpy()
    optimizer_ei = BayesianOptimizer(
        objective = target_new,
        constraint_f = get_utilization,
        constraint = 0.69,
        variables={'a': (0, 27), 'b': (0, 13)},
        n_init=10,
        n_iter=N_ITER,
        acquisition_f='ei',
        batch_size=10
    )
    optimizer_ei.optimize()
    bayes_opt_ei[i, :] = optimizer_ei.best_samples_['y'].to_numpy()
    """
grid_med = np.percentile(-grid_search, 50, axis=0)
grid_lower = np.percentile(-grid_search, 25, axis=0)
grid_upper = np.percentile(-grid_search, 75, axis=0)

#bayes_opt_ucb_med = np.percentile(-bayes_opt_ucb, 50, axis=0)
#bayes_opt_ucb_lower = np.percentile(-bayes_opt_ucb, 25, axis=0)
#bayes_opt_ucb_upper = np.percentile(-bayes_opt_ucb, 75, axis=0)

naive_med = np.percentile(-naive_bayes, 50, axis=0)
naive_lower = np.percentile(-naive_bayes, 25, axis=0)
naive_upper = np.percentile(-naive_bayes, 75, axis=0)
"""
bayes_opt_pi_med = np.percentile(-bayes_opt_pi, 50, axis=0)
bayes_opt_pi_lower = np.percentile(-bayes_opt_pi, 10, axis=0)
bayes_opt_pi_upper = np.percentile(-bayes_opt_pi, 25, axis=0)
bayes_opt_ei_med = np.percentile(-bayes_opt_ei, 50, axis=0)
bayes_opt_ei_lower = np.percentile(-bayes_opt_ei, 1, axis=0)
bayes_opt_ei_upper = np.percentile(-bayes_opt_ei, 25, axis=0)
"""

#plt.plot(np.linspace(0, N_ITER, N_ITER), grid_med, 'C3')
#plt.fill_between(np.linspace(0, N_ITER, N_ITER), grid_lower, grid_upper, color='C3', alpha=0.25)

#plt.plot(np.linspace(0, N_ITER, N_ITER), bayes_opt_ucb_med, color='C0', alpha=1)
#plt.fill_between(np.linspace(0, N_ITER, N_ITER), bayes_opt_ucb_lower, bayes_opt_ucb_upper, color='C0', alpha=0.25)

#plt.plot(np.linspace(0, N_ITER, N_ITER), naive_med, color='C1', alpha=1)
#plt.fill_between(np.linspace(0, N_ITER, N_ITER), naive_lower, naive_upper, color='C1', alpha=0.25)


"""
plt.plot(np.linspace(0, N_ITER, N_ITER), bayes_opt_pi_med, color='C1', alpha=1)
plt.fill_between(np.linspace(0, N_ITER, N_ITER), bayes_opt_pi_lower, bayes_opt_pi_upper, color='C1', alpha=0.25)

plt.plot(np.linspace(0, N_ITER, N_ITER), bayes_opt_ei_med, color='C2', alpha=1)
plt.fill_between(np.linspace(0, N_ITER, N_ITER), bayes_opt_ei_lower, bayes_opt_ei_upper, color='C2', alpha=0.25)
"""
#plt.xlabel('Iteration')
#plt.ylabel('Risk')
#plt.ylim(0, 0.3)
#plt.legend(['Grid Search', '', 'UCB', ''])
#legend_elements = [
#    Patch(facecolor='C3', label='Grid Search'),
#    Patch(facecolor='C0', label='Ours'),
#    Patch(facecolor='C1', label='Naive Bayes'),
#]
# Create the figure
#plt.legend(handles=legend_elements, loc='center')

#plt.show()
np.savez(f'bak{TD}_bayes.npy', bayes_opt_ucb=bayes_opt_ucb)
#np.savez('naive.npy', naive_bayes=naive_bayes)
print(f'ACTUAL_MIN: {np.amin(RISK_OOD_FIRST)}')
