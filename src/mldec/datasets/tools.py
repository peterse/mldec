
import numpy as np
def sample_histogram(probs, m):
    """Sample m bitstrings from a distribution defined by probs.

    This is used for "virtual training": since we have discrete data,
    instead of sampling `n_train` bitstrings for training data, we can
    just reweight a loss function according to a histogram of n_train 
    samples from probs.

    Args:
        probs: (N,) array of probabilities of each bitstring
        m: number of samples to draw
    Returns:
        hist: (N,) array of counts of each bitstring in the sample.
    """
    sample = np.random.choice(len(probs), m, p=probs)
    hist = np.zeros(len(probs))
    for s in sample:
        hist[s] += 1
    return hist / m
