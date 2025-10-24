# inference/conjugate_inference.py
import numpy as np
from scipy.stats import beta
    
class BetaBinomialConjugate:
    """Beta-Binomial conjugate inference."""

    def __init__(self, metric_summary: dict, prior_alpha=1, prior_beta=1, sampling_size=10000):
        self.successes = metric_summary["successes"]
        self.metric_summary = metric_summary
        self.n = metric_summary["n"]
        self.sampling_size = sampling_size
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def sample_posterior(self):
        a_post = self.prior_alpha + self.metric_summary["successes"]
        b_post = self.prior_beta + self.metric_summary["n"] - self.metric_summary["successes"]
        self.posterior_samples = beta(a_post, b_post).rvs(size=self.sampling_size)
        return self.posterior_samples
    
class NormalNormalConjugate:
    def __init__(self, metric_summary, prior_mean=0, prior_var=1, sampling_size=10000):
        self.metric_summary = metric_summary
        self.sampling_size = sampling_size
        self.prior_mean = prior_mean
        self.prior_var = prior_var

    def sample_posterior(self):
        n = self.metric_summary["n"]
        x_bar = self.metric_summary["mean"]
        sigma2 = self.metric_summary["var"]
        post_var = 1 / (n/sigma2 + 1/self.prior_var)
        post_mean = post_var * (n*x_bar/sigma2 + self.prior_mean/self.prior_var)
        self.posterior_samples = np.random.normal(post_mean, np.sqrt(post_var), self.sampling_size)
        return self.posterior_samples
    

