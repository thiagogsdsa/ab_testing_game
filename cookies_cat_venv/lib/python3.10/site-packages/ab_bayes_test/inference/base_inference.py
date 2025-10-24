# inference/base_inference.py
from abc import ABC, abstractmethod
import numpy as np

class BaseInference(ABC):
    """Interface for Bayesian inference."""

    def __init__(self, metric_summary: dict, sampling_size: int = 10000):
        self.metric_summary = metric_summary
        self.sampling_size = sampling_size
        self.posterior_samples = None

    @abstractmethod
    def sample_posterior(self) -> np.ndarray:
        """Generate posterior samples."""
        pass
