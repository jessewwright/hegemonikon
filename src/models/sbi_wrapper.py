import numpy as np
import torch
from sbi import utils as sbi_utils
from sbi.inference import SNPE_C

class SBINESEstimator:
    def __init__(self, simulator, prior_params):
        self.simulator = simulator
        self.prior = sbi_utils.BoxUniform(
            low=np.array([p['low'] for p in prior_params]),
            high=np.array([p['high'] for p in prior_params])
        )
        self.inference = SNPE_C(device='cpu')

    def train_posterior(self, num_simulations=1000):
        theta = self.prior.sample((num_simulations,))
        x = torch.stack([self.simulator(t) for t in theta])
        density_estimator = self.inference.append_simulations(theta, x).train()
        return self.inference.build_posterior(density_estimator)
