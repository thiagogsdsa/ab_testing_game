import numpy as np 
from scipy.stats import norm

def bernoulli_log_likelihood(theta, data):
    if theta <= 0 or theta >= 1:
        return -np.inf
    return np.sum(data * np.log(theta) + (1 - data) * np.log(1 - theta))

def normal_log_likelihood(theta, data):
     return np.sum(norm.logpdf(data, loc=theta, scale=1))