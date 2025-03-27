import os
import numpy as np

import itertools
from mldec.codes import toric_code, code_utils
from mldec.utils import bit_tools
from mldec.datasets import tools
import torch

# get abspath of the directory containing this module
abspath = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(abspath, "cache")

def config_to_fname(n, config, only_good_examples=False):
    """Caching utility.

    warning: don't use config values with more than 3 significant digits.
    """
    ### FIXME: no more alpha?
    only_good = ""
    if only_good_examples:
        only_good = "_only_good"
    return f"toric_code_n{n}_vardepol_p{config['p']:4.3f}_var{config['var']:4.3f}_beta{config['beta']:4.3f}.pt"


def try_to_load_otherwise_make(fname):
    """Automate everything."""
    global CACHE
    if not os.path.exists(CACHE):
        os.makedirs(CACHE)
    path = os.path.join(CACHE, fname)
    if os.path.exists(path):
        probs = torch.load(path)
        X = torch.load(os.path.join(CACHE, "X.pt"))
        Y = torch.load(os.path.join(CACHE, "Y.pt"))
        return X, Y, probs
    return None


def cache_data(X, Y, probs, fname):
    global CACHE
    path = os.path.join(CACHE, fname)
    torch.save(probs, path, _use_new_zipfile_serialization=False)
    torch.save(X, os.path.join(CACHE, "X.pt"), _use_new_zipfile_serialization=False)
    torch.save(Y, os.path.join(CACHE, "Y.pt"), _use_new_zipfile_serialization=False)


def uniform_over_good_examples(n, config, cache=True):
    """Create a uniform distribution over good examples in the toric code
    config contents:

    Returns:
        X: (2**(n-1), 2**2, n-1) array of syndrome bitstrings, indexed by (syndrome, logical)
        Y: (2**(n-1), 2**2, 2) array of logical bitstrings, indexed by (syndrome, logical)
        p_TL: (2**(n-1), 2**2) array of probabilities uniform over good examples.
    """

    # build the p_TL table of coset probabilities 
    X, Y, p_TL = create_dataset_training(n, config, cache=cache)
    # for only-good-examples, we keep just the maximum-probability logical error
    # breaking ties arbitrarily. numpy breaks ties according to the first-occuring max value
    p_TL = p_TL.reshape(-1, 2**2)
    max_indices = p_TL.argmax(1)
    probs = np.zeros_like(p_TL)
    probs[np.arange(p_TL.shape[0]), max_indices] = 1
    probs = probs / probs.sum()
    probs = probs.reshape(-1)
    X, Y, probs = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32), torch.tensor(probs, dtype=torch.float32)
    # note that sos/eos is taken care of by the training set
    return X, Y, probs


def create_dataset_training(n, config, cache=True):
    """Create a set of 'true' weights to be sampled for a training set.

    config contents:

    Returns:
        X: (2**(n+1), n-1) array of syndrome bitstrings, indexed by (syndrome, logical)
        Y: (2**(n+1),  2) array of logical bitstrings, indexed by (syndrome, logical)
        p_TL: (2**(n+1)) array of coset probabilities, indexed by (syndrome, logical)
    
    """
    if n != 9:
        raise NotImplementedError("Only L=3 is implemented for now.")
    L = 3

    sos_eos = config.get("sos_eos")
    if cache:
        target = config_to_fname(n, config)
        out = try_to_load_otherwise_make(target)
        if out is not None:
            X, Y, p_TL = out
            # we don't save datasets with eos/sos, that's a waste of time. just reoload them
            if sos_eos:
                sos, eos = sos_eos
                Y = torch.cat([sos*torch.ones((Y.shape[0], 1)), Y, eos*torch.ones((Y.shape[0], 1))], axis=1)
            return X, Y, p_TL
        
    # We start by building the prism. At the same time, we keep track of the output
    # set of syndromes and logicals, in the order that their wieghts are computed.
    generators_S, generators_T, generators_L, Hx, Hz = toric_code.generators_STL_Hx_Hz(L)
    # err_prism = np.zeros((2**(n-1), 2**2, 2*n))
    X = np.zeros((2**(n-1), 2**2, n-1)) # indexed by (syndrome, logical, X)
    Y = np.zeros((2**(n-1), 2**2, 2))

    noise_model = make_variance_noise_model(n, config)
    p_TL = np.zeros((2**(n-1), 2**2))

    for i_sigma, pure_error_lst in enumerate(code_utils.powerset(generators_T)):
        pure_error = code_utils.operator_sequence_to_stim(pure_error_lst, n)
        xerr, zerr = pure_error.to_numpy()
        sigma_z = (Hz @ xerr) % 2
        sigma_x = (Hx @ zerr) % 2
        sigma = np.concatenate((sigma_x, sigma_z), axis=0)
        X[i_sigma,:] = sigma
        logical_masks = np.array(list(itertools.product([0, 1], repeat=2)))
        for j_logical, logical_mask in enumerate(logical_masks):
            # to enumerate over the logicals, we want to preserve their order 
            # as provided in generators_L, as this will determine the binary
            # mask representing that logical
            logical_op_lst = [generators_L[i] for i, bit in enumerate(logical_mask) if bit]
            logical_op = code_utils.operator_sequence_to_stim(logical_op_lst, n)
            Y[i_sigma, j_logical] = np.array(logical_mask)
            
            for k_stab, stabilizer_lst in enumerate(code_utils.powerset(generators_S)):
                stabilizer = code_utils.operator_sequence_to_stim(stabilizer_lst, n)
                error = pure_error * logical_op * stabilizer
                error_symplectic = np.concatenate(error.to_numpy(), axis=0).astype(int)
                p_err = noise_model(error_symplectic, n)
                p_TL[i_sigma, j_logical] += p_err
                # FIXME: this would be way faster (4x?) if i vectorized, but i'm caching everything anyways.
                # err_prism[i_sigma, j_logical, :] = error_symplectic

    # FIXME: more efficient with vectorization
    # generate the shape (2**(n-1), 2**2) array of coset probabilities
    # p_TL = np.apply_along_axis(noise_model, 1, err_prism.reshape(-1, 2*n), n).reshape(2**(n-1), 2**2)
    # # p_TL = noise_model(err_prism.reshape(-1, 2*n), n).reshape(2**(n-1), 2**2)
    X = X.reshape(-1, n-1)
    Y = Y.reshape(-1, 2)
    p_TL = p_TL.reshape(-1)

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    p_TL = torch.tensor(p_TL, dtype=torch.float32)

    if cache:
        cache_data(X, Y, p_TL, target)
    # only add sos/eos after saving
    if sos_eos:
        sos, eos = sos_eos
        Y = torch.cat([sos*torch.ones((Y.shape[0], 1)), Y, eos*torch.ones((Y.shape[0], 1))], axis=1)
    return X, Y, p_TL


def sample_virtual_XY(probs, m, n, dataset_config, cache=True):
    """Sample virtual dataset.
    
    The virtual dataset is a tuple (X, Y, weights).


    args:
        probs: (N,) array of probabilities of each datum
        m: number of samples to draw
        n: number of qubits
        config: Should contain...
        sos_eos: Tuple (sos, eos) to append to the targets. If None, no tokens are appended.
    Returns:
        Y is (m, 2) array, where N is the number of unique samples and n is the number of bits.
            Each row of Y is a unique bitstring that appeared in sampling.
        X is (m, n-1) array of associated syndromes
        weights: (N,) array of weights for each sample.
        hist: (2**n,) array of counts of each bitstring in the dataset, ordered correctly to match
            the ordering lf `probs`. This is used to build a complete training set iteratively during batching.

    """
    if cache:
        target = config_to_fname(n, dataset_config)
        out = try_to_load_otherwise_make(target)
        if out is None:
            raise ValueError("you haven't cached training data yet.")
        X_full, Y_full, _ = out # shapes (2^(n+1), n-1) and (2^(n+1), 2) respectively
        
    # We're going to work with a 'ragged' array, which contains only bitstrings that were sampled
    # for this dataset
    hist = tools.sample_histogram(probs, m) 

    Y = np.array([Y_full[i] for i in range(len(probs)) if hist[i] > 0])
    X = np.array([X_full[i] for i in range(len(probs)) if hist[i] > 0])
    weights = np.array([hist[i] for i in range(len(probs)) if hist[i] > 0])
    weights = weights / weights.sum()
    Xb_tensor = torch.tensor(X, dtype=torch.float32)
    Yb_tensor = torch.tensor(Y, dtype=torch.float32)
    sos_eos = dataset_config.get("sos_eos")
    if sos_eos:
        sos, eos = sos_eos
        Yb_tensor = torch.cat([sos*torch.ones((Yb_tensor.shape[0], 1)), Yb_tensor, eos*torch.ones((Yb_tensor.shape[0], 1))], axis=1)
    weightsb = torch.tensor(weights, dtype=torch.float32)

    return Xb_tensor, Yb_tensor, weightsb, hist


def make_variance_noise_model(n, config, return_probs=False):
    """
    make a set of probabilities {p_i} with mean p, variance var, and then rescale it by beta.

    For a particular choice of p, var this will deterministically produce error probabilities
    for each qubit. This functionality is intended for running many training experiments on 
    the same underlying 'device'.

    Args:
        n: number of qubits
        config: dictionary with keys 
            'p': average probability
            'var': variance of errors 
            'beta': scaling factor for the entire vector
        return_probs: if True, return the vector of probabilities instead of the noise model.
    """
    assert n == 9 
    p = config.get('p')
    # alpha = config.get('alpha')
    var = config.get('var')
    beta = config.get('beta')
    # Explanation:
    # we want to train many models on the same underlying error model so that the 
    # variance in performance is due to model randomness rather than some weird
    # counterfactual where we compare devices with different noise models.
    # BUT we cannot use np.seed, e.g.
    #    np.random.seed(222)
    #   p_samp = np.random.normal(p, var, size=n)
    # the problem is that this will poison our training set sampling downstream! 
    # this function gets called in a training set sampler, which means we pre-seed
    # a random sampling of training examples from a noise model which is itself
    # nonrandom. The solution is to generate a specific noise model with the 
    # properties I wanted, and then fix that without setting any seed.
    if var == 0:
        p_samp = p * np.ones(n)
    elif (p == 0.05 and var == 0.03):
        # below is the output of the following code:
        # np.random.seed(222)
        # p_samp = np.random.normal(p, var, size=n)
        p_samp = np.array([0.10890275, 0.05827309, 0.06375975, 0.08003794, 0.02708494,
       0.07165783, 0.02283591, 0.0800562 , 0.03437773])
    else:
        raise NotImplementedError("sample first, then hardcode.")
    
    p_samp = p_samp * beta
    if return_probs:
        return p_samp

    def variance_noise_model(err, n):
        """
        After sampling some vector (p_1, ..., p_n) with variance var, each qubit has a different 
         depol. channel probability p_i

        Args:
            error: [error_x, error_z] with length 2n
        """
        is_err = np.logical_or(err[:n], err[n:]).astype(bool)
        is_not_err = np.logical_not(is_err).astype(bool)
        error_prob = np.prod(p_samp[is_err] / 3)
        no_error_prob = np.prod(1-p_samp[is_not_err])
        out = error_prob * no_error_prob
        return out
    
    return variance_noise_model

