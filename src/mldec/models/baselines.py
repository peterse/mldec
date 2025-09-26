import numpy as np 
import torch
from pymatching import Matching
assert int(np.__version__[0]) < 2, "pymatching is not usable with numpy 2"

from mldec.codes import toric_code, steane_code, fivequbit_code
from mldec.utils import bit_tools

class LookupTable():
    """Train a lookup table to just return the most likely error given a syndrome, from empirical data."""
    def __init__(self):
        self.table = {} # map {0,1}^(n-1) -> {0,1}^n
        self.output_len = None
    
    def train_on_histogram(self, X, Y, hist):
        """Train the lookup table on a histogram of (complete) training data.

        Whenever a training point is 'missing', the naive lookup table assumes the error is 0...0.
        This is because any other behavior (other than guessing randomly) would encode
        information about the stabilizer group that we don't assume we have.
        
        X: (2**(n-1),) array of syndromes
        Y: (2**n, ) array of errors
        hist: (2**n, ) histogram of training data counts corresponding to Y
        """
        dct = {}
        self.output_len = len(Y[0])
        if isinstance(X, torch.Tensor):
            X, Y = X.numpy(), Y.numpy()
        for x, y, p in zip(X, Y, hist):
            x = tuple(x)
            y = tuple(y)
            if x not in dct:
                dct[x] = {}
            if y not in dct[x]:
                dct[x][y] = 0
            dct[x][y] = p

        # Now we have a dictionary of counts. For each syndrome, find the most likely error
        for x in dct:
            xkey = tuple(x)
            # find the key in dct[xkey] with the highest value

            yvals = np.array(list(dct[xkey].keys()))
            probs = [dct[xkey][tuple(y)] for y in yvals]
            if max(probs) == 0:
                self.table[xkey] = np.array([0]*self.output_len)
                continue
            max_key = yvals[np.argmax(probs)]
            self.table[xkey] = max_key


    def predict(self, X):
        out = []
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        for x in X:
            ypred = self.table[tuple(x)]
            if ypred is None:
                ypred = np.array([0]*self.output_len)
            out.append(ypred)
        return torch.tensor(out)
    

class RepetitionCodeMinimumWeight():
    """Evaluate the 'minimum weight decoder' for a repetition code."""
    def __init__(self):
        self.table = {}

    def make_decoder(self, X, Y):
        """This does not train the decoder, rather we are just re-using syndrome-error pairs"""
        if isinstance(X, torch.Tensor):
            X, Y = X.numpy(), Y.numpy()
        for (x, y) in zip(X, Y):
            xkey = tuple(x)
            if self.table.get(xkey) is None:
                self.table[xkey] = y
            else:
                # get the minimum weight error
                if sum(y) < sum(self.table[xkey]):
                    self.table[xkey] = y

    def predict(self, X):
        out = []
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        for x in X:
            out.append(self.table[tuple(x)])
        return torch.tensor(out)


class MinimumWeightPerfectMatching():
    """Compute MWPM for an input toric syndrome(s)"""

    def __init__(self, L=3):
        self.L = L
        self.n = L ** 2
        self.generators_L = None
        self.lookup = None

    def make_decoder(self, X, Y, error_probs=None):
        _, _, Hx, Hz = toric_code.rotated_surface_code_stabilizers(self.L)
        Hx = torch.tensor(Hx)
        Hz = torch.tensor(Hz)
        weights = None
        if error_probs is not None:
            weights = np.log( (1- error_probs) / error_probs)
        self.matching_x = Matching(Hx, weights=weights)
        self.matching_z = Matching(Hz, weights=weights)
        # to decode in the DDD setting we need a lookup table mapping errors to logicals
        # (or you can implement gaussian elim, whatever).
        self.lookup = toric_code.build_lst_lookup(self.L, cache=True)

    def predict(self, X):
        """
        
        Args:
            X: shape (N, n-1) array of syndromes, in concatenated [sigma_x, sigma_z] format
        """
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        sigma_x = X[:,:(self.n - 1)//2] # detections of Z-type events, i.e. x stabilizers
        sigma_z = X[:,(self.n - 1)//2:] # detections of X-type events

        # predict the physical errors
        xerr_pred = self.matching_z.decode_batch(np.array(sigma_z))
        zerr_pred = self.matching_x.decode_batch(np.array(sigma_x))
        error_preds = np.concatenate((xerr_pred.reshape(-1, self.n), zerr_pred.reshape(-1, self.n)), axis=1)
        error_idx = bit_tools.bits_to_ints(error_preds)
        # get the logical operator such that \ell*S contains this error
        sigma_logical = self.lookup[error_idx]
        Ypred = sigma_logical[:, self.n-1:]
        return torch.tensor(Ypred)


# class SteaneDecoder():
#     """Compute MWPM for an input toric syndrome(s)"""

#     def __init__(self, L=3):
#         self.n = 7
#         self.generators_L = None
#         self.lookup = None

#     def make_decoder(self, X, Y, error_probs=None):
#         _, _, Hx, Hz = steane_code.generators_STL_Hx_Hz(self.n)
#         Hx = torch.tensor(Hx)
#         Hz = torch.tensor(Hz)
#         weights = None
#         if error_probs is not None:
#             weights = np.log( (1- error_probs) / error_probs)
#         self.matching_x = Matching(Hx, weights=weights)
#         self.matching_z = Matching(Hz, weights=weights)
#         # to decode in the DDD setting we need a lookup table mapping errors to logicals
#         # (or you can implement gaussian elim, whatever).
#         self.lookup = toric_code.build_lst_lookup(self.L, cache=True)

#     def predict(self, X):
#         """
        
#         Args:
#             X: shape (N, n-1) array of syndromes, in concatenated [sigma_x, sigma_z] format
#         """
#         if isinstance(X, torch.Tensor):
#             X = X.numpy()
#         sigma_x = X[:,:(self.n - 1)//2] # detections of Z-type events, i.e. x stabilizers
#         sigma_z = X[:,(self.n - 1)//2:] # detections of X-type events

#         # predict the physical errors
#         xerr_pred = self.matching_z.decode_batch(np.array(sigma_z))
#         zerr_pred = self.matching_x.decode_batch(np.array(sigma_x))
#         error_preds = np.concatenate((xerr_pred.reshape(-1, self.n), zerr_pred.reshape(-1, self.n)), axis=1)
#         error_idx = bit_tools.bits_to_ints(error_preds)
#         # get the logical operator such that \ell*S contains this error
#         sigma_logical = self.lookup[error_idx]
#         Ypred = sigma_logical[:, self.n-1:]
#         return torch.tensor(Ypred)



class CyclesMinimumWeightPerfectMatching():

    def __init__(self, detector_error_model,L=3):
        self.L = L
        self.n = L ** 2
        self.matching = Matching.from_detector_error_model(detector_error_model)

    def predict(self, X):
        """Make predictions for observable flips given a list of detection events.
        
        Argse:
            X: list of detection events, i.e. first returned value of sampler.sample
        """
        return self.matching.decode_batch(X)

