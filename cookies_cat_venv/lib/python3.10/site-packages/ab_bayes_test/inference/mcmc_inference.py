import numpy as np
from scipy.stats import beta, norm



def safe_log(x, eps=1e-12):
    """Safe log to avoid -inf."""
    return np.log(x + eps)

import numpy as np

def safe_log(x, eps=1e-12):
    return np.log(x + eps)

class MCMCInference:
    """
    Metropolis-Hastings MCMC sampler for Bayesian inference.

    Parameters:
        data (np.ndarray): raw data points (individual observations)
        prior_func (callable): function returning a dict with 'rvs' and 'pdf'
        likelihood_func (callable): function (theta, data) -> log-likelihood
        sampling_size (int): number of posterior samples
        proposal_sd (float): standard deviation for proposal distribution
        support (tuple): optional (min, max) support for theta
    """
    def __init__(self, data, prior_func, likelihood_func, sampling_size=10000, proposal_sd=0.05, support=None):
        self.data = data
        self.prior_func = prior_func
        self.likelihood_func = likelihood_func
        self.sampling_size = sampling_size
        self.proposal_sd = proposal_sd
        self.support = support
        self.posterior_samples = None

    def sample_posterior(self):
        samples = []

        # initialize at prior
        prior = self.prior_func()
        current = prior["rvs"]()

        log_prior_current = safe_log(prior["pdf"](current))
        log_likelihood_current = self.likelihood_func(current, self.data)

        for _ in range(self.sampling_size):
            # propose new value
            proposal = current + np.random.normal(0, self.proposal_sd)

            # reject if outside support
            if self.support is not None:
                if proposal < self.support[0] or proposal > self.support[1]:
                    samples.append(current)
                    continue

            # compute log-prior and log-likelihood for proposal
            log_prior_proposal = safe_log(prior["pdf"](proposal))
            log_likelihood_proposal = self.likelihood_func(proposal, self.data)

            # acceptance probability
            alpha = min(1, np.exp((log_prior_proposal + log_likelihood_proposal) -
                                  (log_prior_current + log_likelihood_current)))
            if np.random.rand() < alpha:
                current = proposal
                log_prior_current = log_prior_proposal
                log_likelihood_current = log_likelihood_proposal

            samples.append(current)

        self.posterior_samples = np.array(samples)
        return self.posterior_samples

# class MCMCInference:
#     def __init__(self, summary, prior_func, likelihood_func, sampling_size=10000, proposal_sd=0.05, support=None):
#         self.summary = summary
#         self.prior_func = prior_func
#         self.likelihood_func = likelihood_func
#         self.sampling_size = sampling_size
#         self.proposal_sd = proposal_sd
#         self.support = support
#         self.posterior_samples = None

#     def sample_posterior(self):
#         samples = []

#         # inicializa no prior
#         prior = self.prior_func()
#         current = prior["rvs"]()

#         log_prior_current = safe_log(prior["pdf"](current))
#         log_likelihood_current = self.likelihood_func(current, self.summary)

#         for _ in range(self.sampling_size):
#             proposal = current + np.random.normal(0, self.proposal_sd)

#             # rejeita fora do suporte
#             if self.support is not None:
#                 if proposal < self.support[0] or proposal > self.support[1]:
#                     samples.append(current)
#                     continue

#             prior = self.prior_func()
#             log_prior_proposal = safe_log(prior["pdf"](proposal))
#             log_likelihood_proposal = self.likelihood_func(proposal, self.summary)

#             alpha = min(1, np.exp((log_prior_proposal + log_likelihood_proposal) - 
#                                   (log_prior_current + log_likelihood_current)))
#             if np.random.rand() < alpha:
#                 current = proposal
#                 log_prior_current = log_prior_proposal
#                 log_likelihood_current = log_likelihood_proposal

#             samples.append(current)

#         self.posterior_samples = np.array(samples)
#         return self.posterior_samples
    
# class MCMCInference:
#     """
#     Generic Metropolis-Hastings sampler for 1D posteriors.
#     Works for both proportion (0-1) and real-valued mean metrics.
    
#     prior_func: callable returning dict {"rvs":..., "pdf":...}
#     likelihood_func: callable(theta, summary) -> density
#     support: tuple (min, max) or None for unbounded
#     """
#     def __init__(self, summary, prior_func, likelihood_func,
#                  sampling_size=10000, proposal_sd=None, support=None):
#         self.summary = summary
#         self.prior_func = prior_func
#         self.likelihood_func = likelihood_func
#         self.sampling_size = sampling_size
#         self.support = support
        
#         # Adaptive proposal_sd if not provided
#         if proposal_sd is None:
#             if support is not None:  # proportion
#                 self.proposal_sd = 0.05  # small step
#             else:  # mean: scale by standard error
#                 self.proposal_sd = np.sqrt(summary["var"] / summary["n"])
#         else:
#             self.proposal_sd = proposal_sd

#         self.posterior_samples = None

#     def sample_posterior(self):
#         samples = []
        
#         # initialize near posterior mode
#         current = self.prior_func()["rvs"]()
#         log_prior_current = safe_log(self.prior_func()["pdf"](current))
#         log_likelihood_current = safe_log(self.likelihood_func(current, self.summary))
        
#         for _ in range(self.sampling_size):
#             proposal = current + np.random.normal(0, self.proposal_sd)
            
#             # enforce support
#             if self.support is not None:
#                 proposal = np.clip(proposal, self.support[0], self.support[1])
            
#             log_prior_proposal = safe_log(self.prior_func()["pdf"](proposal))
#             log_likelihood_proposal = safe_log(self.likelihood_func(proposal, self.summary))
            
#             # acceptance probability
#             alpha = min(1, np.exp(
#                 (log_prior_proposal + log_likelihood_proposal) -
#                 (log_prior_current + log_likelihood_current)
#             ))
            
#             if np.random.rand() < alpha:
#                 current = proposal
#                 log_prior_current = log_prior_proposal
#                 log_likelihood_current = log_likelihood_proposal
            
#             samples.append(current)
        
#         self.posterior_samples = np.array(samples)
#         return self.posterior_samples


# class MCMCInference:
#     """
#     Generic 1D Metropolis-Hastings sampler for Bayesian inference.
#     Supports probability parameters (0,1) and real-valued parameters (e.g., means).
    
#     Parameters
#     ----------
#     metric_summary : dict
#         Summary of the data. For proportion: {"successes": s, "n": n}.
#         For mean: {"mean": m, "var": v, "n": n}.
#     prior_func : callable
#         Function returning dict with "rvs" and "pdf" keys.
#     likelihood_func : callable
#         Function(theta, metric_summary) returning likelihood.
#     sampling_size : int
#         Number of posterior samples.
#     proposal_sd : float
#         Standard deviation of proposal distribution.
#     support : tuple or None
#         Valid parameter range. For probability, use (0,1). For mean, None.
#     """
    
#     def __init__(self, metric_summary, prior_func, likelihood_func,
#                  sampling_size=10000, proposal_sd=None, support=None):
#         self.metric_summary = metric_summary
#         self.prior_func = prior_func
#         self.likelihood_func = likelihood_func
#         self.sampling_size = sampling_size
#         self.support = support
        
#         # Automatic proposal scaling for mean metric
#         if proposal_sd is None:
#             if "var" in metric_summary:  # mean metric
#                 self.proposal_sd = np.sqrt(metric_summary["var"]) / 2
#             else:  # proportion
#                 self.proposal_sd = 0.05
#         else:
#             self.proposal_sd = proposal_sd
        
#     def sample_posterior(self):
#         samples = []
        
#         # Initialize at prior sample
#         current = self.prior_func()["rvs"]()
#         log_prior_current = safe_log(self.prior_func()["pdf"](current))
#         log_likelihood_current = safe_log(self.likelihood_func(current, self.metric_summary))
        
#         for _ in range(self.sampling_size):
#             # Propose candidate
#             proposal = current + np.random.normal(0, self.proposal_sd)
            
#             # Reject if outside support
#             if self.support is not None:
#                 low, high = self.support
#                 if proposal < low or proposal > high:
#                     samples.append(current)
#                     continue
            
#             # Compute acceptance
#             log_prior_proposal = safe_log(self.prior_func()["pdf"](proposal))
#             log_likelihood_proposal = safe_log(self.likelihood_func(proposal, self.metric_summary))
            
#             alpha = min(1, np.exp((log_prior_proposal + log_likelihood_proposal) -
#                                   (log_prior_current + log_likelihood_current)))
            
#             if np.random.rand() < alpha:
#                 current = proposal
#                 log_prior_current = log_prior_proposal
#                 log_likelihood_current = log_likelihood_proposal
            
#             samples.append(current)
        
#         self.posterior_samples = np.array(samples)
#         return self.posterior_samples


# class MCMCInference:
#     """
#     Generic Metropolis-Hastings sampler for 1D posteriors.
    
#     Prior_func: callable returning {"rvs": ..., "pdf": ...}
#     Likelihood_func: callable(theta, metric_summary) -> probability density
#     support: tuple (min, max) for valid theta values. Use None for unbounded (mean case)
#     """
#     def __init__(self, metric_summary, prior_func, likelihood_func,
#                  sampling_size=10000, proposal_sd=0.05, support=None):
#         self.metric_summary = metric_summary
#         self.prior_func = prior_func
#         self.likelihood_func = likelihood_func
#         self.sampling_size = sampling_size
#         self.proposal_sd = proposal_sd
#         self.support = support  # e.g., (0,1) for proportion; None for real numbers

#     def sample_posterior(self):
#         samples = []

#         # initialize at prior sample
#         current = self.prior_func()["rvs"]()
#         log_prior_current = safe_log(self.prior_func()["pdf"](current))
#         log_likelihood_current = safe_log(self.likelihood_func(current, self.metric_summary))

#         for _ in range(self.sampling_size):
#             # propose new candidate
#             proposal = current + np.random.normal(0, self.proposal_sd)

#             # reject outside valid support
#             if self.support is not None:
#                 if proposal < self.support[0] or proposal > self.support[1]:
#                     samples.append(current)
#                     continue

#             log_prior_proposal = safe_log(self.prior_func()["pdf"](proposal))
#             log_likelihood_proposal = safe_log(self.likelihood_func(proposal, self.metric_summary))

#             alpha = min(1, np.exp(
#                 (log_prior_proposal + log_likelihood_proposal) -
#                 (log_prior_current + log_likelihood_current)
#             ))

#             # accept/reject
#             if np.random.rand() < alpha:
#                 current = proposal
#                 log_prior_current = log_prior_proposal
#                 log_likelihood_current = log_likelihood_proposal

#             samples.append(current)

#         self.posterior_samples = np.array(samples)
#         return self.posterior_samples


# class MCMCInference:
#     """
#     Generic Metropolis-Hastings sampler for 1D posteriors.
#     Prior_func: callable returning initial sample and pdf
#     Likelihood_func: callable(theta, metric_summary) -> probability density
#     """
#     def __init__(self, metric_summary, prior_func, likelihood_func, sampling_size=10000, proposal_sd=0.05):
#         self.metric_summary = metric_summary
#         self.prior_func = prior_func
#         self.likelihood_func = likelihood_func
#         self.sampling_size = sampling_size
#         self.proposal_sd = proposal_sd

#     def sample_posterior(self):
#         samples = []
        
#         # initialize at prior sample
#         current = self.prior_func()["rvs"]()
        
#         log_prior_current = safe_log(self.prior_func()["pdf"](current))
#         log_likelihood_current = safe_log(self.likelihood_func(current, self.metric_summary))
        
#         for _ in range(self.sampling_size):
#             # propose new candidate
#             proposal = current + np.random.normal(0, self.proposal_sd)
            
#             # optional: reject outside valid support
#             if proposal < 0 or proposal > 1:
#                 samples.append(current)
#                 continue

#             log_prior_proposal = safe_log(self.prior_func()["pdf"](proposal))
#             log_likelihood_proposal = safe_log(self.likelihood_func(proposal, self.metric_summary))
#             alpha = min(1, np.exp((log_prior_proposal + log_likelihood_proposal) - 
#                                   (log_prior_current + log_likelihood_current)))
            
#             if np.random.rand() < alpha:
#                 current = proposal
#                 log_prior_current = log_prior_proposal
#                 log_likelihood_current = log_likelihood_proposal

#             samples.append(current)

#         self.posterior_samples = np.array(samples)
#         return self.posterior_samples


# import numpy as np
# # import pymc as pm

# import numpy as np
# from .base_inference import BaseInference

# class MCMCInference(BaseInference):
#     """
#     Generic MCMC inference class.
#     Prior and likelihood are provided as functions.
#     Uses simple Metropolis-Hastings sampling.
#     """

#     def __init__(self, metric_summary: dict, prior_func, likelihood_func, sampling_size=10000, proposal_sd=0.1):
#         super().__init__(metric_summary, sampling_size)
#         self.prior_func = prior_func
#         self.likelihood_func = likelihood_func
#         self.proposal_sd = proposal_sd

#     def sample_posterior(self):
#         samples = []
#         # initialize at prior mean
#         current = self.prior_func().rvs()
        
#         for _ in range(self.sampling_size):
#             # propose new candidate
#             proposal = current + np.random.normal(0, self.proposal_sd)
            
#             # reject if outside [0,1] (if parameter is probability)
#             if proposal < 0 or proposal > 1:
#                 samples.append(current)
#                 continue
            
#             # compute acceptance ratio
#             numerator = self.likelihood_func(proposal, self.metric_summary) * self.prior_func().pdf(proposal)
#             denominator = self.likelihood_func(current, self.metric_summary) * self.prior_func().pdf(current)
#             alpha = min(1, numerator / denominator)
            
#             # accept/reject
#             if np.random.rand() < alpha:
#                 current = proposal
                
#             samples.append(current)
        
#         self.posterior_samples = np.array(samples)
#         return self.posterior_samples


# class MCMCInference(BaseInference):
#     def __init__(self, metric_summary: dict, prior_func, likelihood_func, sampling_size=10000):
#         super().__init__(metric_summary, sampling_size)
#         self.prior_func = prior_func         # function that returns prior distribution
#         self.likelihood_func = likelihood_func # function that returns likelihood
       
#     def sample_posterior(self):
#         with pm.Model() as model:
#             theta = self.prior_func()
#             self.likelihood_func(theta, self.metric_summary)
#             trace = pm.sample(self.sampling_size, chains=1, cores=1, progressbar=False)
#         self.posterior_samples = trace.posterior.values.flatten()
#         return self.posterior_samples
    

# class MCMCInference(BaseInference):
#     def __init__(self, metric_summary: dict, prior_func, likelihood_func, sampling_size=10000):
#         super().__init__(metric_summary, sampling_size)
#         self.prior_func = prior_func
#         self.likelihood_func = likelihood_func

#     def sample_posterior(self):
#         samples = []
#         current = self.prior_func().rvs()
#         for _ in range(self.sampling_size):
#             proposal = current + np.random.normal(0, 0.1)
#             accept_ratio = (self.likelihood_func(proposal, self.metric_summary) *
#                             self.prior_func().pdf(proposal)) / \
#                            (self.likelihood_func(current, self.metric_summary) *
#                             self.prior_func().pdf(current))
#             if np.random.rand() < accept_ratio:
#                 current = proposal
#             samples.append(current)
#         self.posterior_samples = np.array(samples)
#         return self.posterior_samples