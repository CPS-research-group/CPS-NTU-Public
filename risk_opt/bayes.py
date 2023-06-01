from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from scipy.optimize import minimize
import sys
import pandas as pd
import numpy as np

class BayesianOptimizer():
    def __init__(self, target_func, x_init, y_init, n_iter, scale, batch_size):
        self.x_init = x_init
        self.y_init = y_init
        self.target_func = target_func
        self.n_iter = n_iter
        self.scale = scale
        self.batch_size = batch_size
        self.gauss_pr = GaussianProcessRegressor()
        self.best_samples_ = pd.DataFrame(columns = ['x', 'y', 'ei'])
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

    def _acquisition_function(self, x):
        return -self._get_expected_improvement(x).ravel()

    def _get_next_probable_point(self):
        min_ei = float(sys.maxsize)
        x_optimal = None

        for x_start in (np.random.random((self.batch_size, self.x_init.shape[1])) * self.scale):
            response = minimize(fun=self._acquisition_function, x0 =x_start, method='L-BFGS-B')
            if response.fun[0] < min_ei:
                min_ei = response.fun[0]
                x_optimal = response.x

        return x_optimal, min_ei

    def _extend_prior_with_posterior_data(self, x, y):
        self.x_init = np.append(self.x_init, np.array([x]), axis=0)
        self.y_init = np.append(self.y_init, np.array(y), axis=0)

    def optimize(self):
        y_max_ind = np.argmax(self.y_init)
        y_max = self.y_init[y_max_ind]
        optimal_x = self.x_init[y_max_ind]
        optimal_ei = None
        for i in range(self.n_iter):
            self.gauss_pr.fit(self.x_init, self.y_init)
            x_next, ei = self._get_next_probable_point()
            x_next = np.clip(x_next, [0,0,0,0], [99,27,99,13])
            y_next = self.target_func(np.array([x_next]))
            self._extend_prior_with_posterior_data(x_next,y_next)

            if y_next[0] > y_max:
                y_max = y_next[0]
                optimal_x = x_next
                optimal_ei = ei

            if i == 0:
                prev_x = x_next

            else:
                self.distances_.append(np.linalg.norm(prev_x - x_next))
                prev_x = x_next
            
            self.best_samples_ = self.best_samples_.append({"y": y_max, "ei": optimal_ei}, ignore_index=True)
        return optimal_x, y_max


RISK = np.load('risk_0.333333.npy')
def target(x):
    #if len(x.shape) == 1:
    #    print(x)
    #    return -RISK[int(x[0]), int(x[1]), int(x[2]), int(x[3])]
    out = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        out[i] = -RISK[int(x[i, 0]), int(x[i, 1]), int(x[i, 2]), int(x[i, 3])]
    return out

LEN = 5
x_start = np.zeros((LEN,4))
x_start[:,0] = np.random.randint(100, size=LEN)
x_start[:,1] = np.random.randint(28, size=LEN)
x_start[:,2] = np.random.randint(100, size=LEN)
x_start[:,3] = np.random.randint(14, size=LEN)
y_start = target(x_start)



optimizer = BayesianOptimizer(
    target_func = target,
    x_init =x_start,
    y_init =y_start,
    n_iter =50,
    scale = 1,
    batch_size=1
)
optimizer.optimize()
print(optimizer.best_samples_)