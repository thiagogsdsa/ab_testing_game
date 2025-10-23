
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import beta as beta_dist
from .inference.conjugate_inference import BetaBinomialConjugate, NormalNormalConjugate
from .inference.mcmc_inference import MCMCInference
from .likelihoods.likelihood import bernoulli_log_likelihood,normal_log_likelihood

class ABBayesTest:
    def __init__(self, df: pd.DataFrame, group_col: str, value_col: str,
                 metric_type: str = "proportion", inference_type: str = "conjugate",
                 prior_params=None, sampling_size: int = 10000,
                 control_group=None, treatment_group=None):
        """
        Bayesian A/B test for proportion or mean metrics.

        Parameters:
            df: DataFrame containing the data
            group_col: Column name for group labels (control/treatment)
            value_col: Column name for metric values
            metric_type: 'proportion' or 'mean'
            inference_type: 'conjugate' or 'mcmc'
            prior_params: Dict with prior parameters
            sampling_size: Number of posterior samples
            control_group: Label of the control group
            treatment_group: Label of the treatment group
        """
        if not control_group or not treatment_group:
            raise ValueError("Both control_group and treatment_group must be specified")
        
        self.df = df
        self.group_col = group_col
        self.value_col = value_col
        self.metric_type = metric_type
        self.inference_type = inference_type
        self.prior_params = prior_params or {}
        self.sampling_size = sampling_size
        self.control_group = control_group
        self.treatment_group = treatment_group
        self.groups = [control_group, treatment_group]

        self._distributions = {}
        self._fitted = False

    def _compute_summary(self):
        summaries = {}
        for group in self.groups:
            data = self.df[self.df[self.group_col] == group][self.value_col].values
            if self.metric_type == "proportion":
                successes = np.sum(data)
                n = len(data)
                summaries[group] = {"successes": successes, "n": n}
            else:
                summaries[group] = {"mean": np.mean(data),
                                    "var": np.var(data, ddof=1),
                                    "n": len(data)}
        return summaries

    def fit(self):
        summaries = self._compute_summary()

        for group in self.groups:
            summary = summaries[group]

            if self.metric_type == "proportion":
                alpha_prior = self.prior_params.get("alpha", 1)
                beta_prior = self.prior_params.get("beta", 1)

                if self.inference_type == "conjugate":

                    bb_inference = BetaBinomialConjugate(
                    metric_summary=summary,          
                    prior_alpha=alpha_prior,  
                    prior_beta=beta_prior,   
                    sampling_size=self.sampling_size
                    )
                    samples = bb_inference.sample_posterior()

                else:
                    # MCMC
                    data_points = self.df[self.df[self.group_col] == group][self.value_col].values

                    def beta_prior_func():
                        a_post = alpha_prior + summary["successes"]
                        b_post = beta_prior + summary["n"] - summary["successes"]
                        dist = beta_dist(a_post, b_post)
                        return {"rvs": lambda size=1: dist.rvs(size=size),
                                "pdf": lambda theta: dist.pdf(theta)}
                    
                    prior_func = beta_prior_func
                    likelihood_func = bernoulli_log_likelihood 
 
                    inf = MCMCInference(data_points, prior_func, likelihood_func,
                                        sampling_size=self.sampling_size,
                                        proposal_sd=0.02,
                                        support=(0, 1))
                    samples = inf.sample_posterior()

            else:  # metric_type == "mean"
                data_points = self.df[self.df[self.group_col] == group][self.value_col].values
                data_mean = np.mean(data_points)
                data_std = np.std(data_points, ddof=1)
                data_normalized = (data_points - data_mean) / data_std
                prior_mean = self.prior_params.get("mean", 0)
                prior_var = self.prior_params.get("var", 1)

                if self.inference_type == "conjugate":

                    nn_inference = NormalNormalConjugate(
                            metric_summary=summary,        
                            prior_mean=prior_mean, 
                            prior_var=prior_var,    
                            sampling_size=self.sampling_size
                        )
                    samples = nn_inference.sample_posterior()

                else:
                    support = (-5, 5)
                    proposal_sd = 0.1 

                    def normal_prior_func():
                        return {"rvs": lambda size=1: np.random.normal(prior_mean, np.sqrt(prior_var), size=size),
                                "pdf": lambda theta: norm.pdf(theta, loc=prior_mean, scale=np.sqrt(prior_var))}

                    prior_func = normal_prior_func
                    likelihood_func = normal_log_likelihood
                    inf = MCMCInference(
                                        data_normalized, 
                                        prior_func, 
                                        likelihood_func,
                                        sampling_size=self.sampling_size,
                                        proposal_sd=proposal_sd,
                                        support=support
                                        )
                    samples_normalized = inf.sample_posterior()
                    #Transform samples back to original scale
                    samples = samples_normalized * data_std + data_mean

            self._distributions[group] = samples

        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Fit the test first with .fit()")

    def results(self, ci=None):
        self._check_fitted()
        output = {}
        for group in self.groups:
            samples = self._distributions[group]
            summary = {"mean": np.mean(samples), "std": np.std(samples)}
            if ci is not None:
                lower = np.percentile(samples, (1 - ci) / 2 * 100)
                upper = np.percentile(samples, (1 + ci) / 2 * 100)
                summary.update({"ci_lower": lower, "ci_upper": upper})
            output[group] = summary
        return output

    def lift_summary(self):
        self._check_fitted()
        lift_samples = self._distributions[self.treatment_group] / self._distributions[self.control_group] - 1
        prob_treatment_superior = np.mean(self._distributions[self.treatment_group] > self._distributions[self.control_group])
        return {"mean_lift": np.mean(lift_samples),
                "std_lift": np.std(lift_samples),
                "prob_treatment_superior": prob_treatment_superior}

    def probability(self):
        return self.lift_summary()["prob_treatment_superior"]

    def get_distributions(self):
        self._check_fitted()
        return self._distributions.copy()
