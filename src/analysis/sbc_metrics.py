import numpy as np
from scipy import stats

def compute_rank_statistics(prior_samples, posterior_samples):
    ranks = []
    for i in range(prior_samples.shape[1]):
        ecdf = stats.ecdf(posterior_samples[:,i]).cdf.evaluate
        ranks.append(ecdf(prior_samples[:,i]))
    return np.array(ranks)
