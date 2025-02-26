import numpy as np 
import torch


class RepetitionCodeLookupTable():
    """Train a lookup table to just return the most likely error given a syndrome, from empirical data."""
    def __init__(self):
        self.table = {} # map {0,1}^(n-1) -> {0,1}^n
    
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
            y1, y2 = dct[xkey].keys()
            p1, p2 = dct[xkey][y1], dct[xkey][y2]
            if p1 == p2:
                if p1 == 0:
                    self.table[xkey] = tuple([0]*len(y1))
                    continue
                max_key = (y1, y2)[np.random.choice(2)] # probably never happens
            elif dct[xkey][y1] > dct[xkey][y2]:
                max_key = y1
            else:
                max_key = y2
            self.table[xkey] = max_key
        
    def predict(self, X):
        out = []
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        for x in X:
            out.append(self.table[tuple(x)])
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
