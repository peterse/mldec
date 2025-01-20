import itertools
import numpy as np

def xlogx(x):
    """Safely compute x*logx in base 2"""
    temp = np.log2(x, out=np.zeros_like(x), where=(x!=0))
    return np.multiply(x, temp)


def shannon_entropy(p):
    """Compute the shannon entropy of an arbitrary distribution p"""
    return -np.sum(xlogx(p))


def binary_entropy(p):
    """Compute the binary entropy function H(p) = -p*log(p) - (1-p)*log(1-p)"""
    return - xlogx(p) - xlogx(1 - p)

def inverse_binary_entropy(y):
    """approximate inverse of binary entropy"""
    return 0.5 * (1 - np.sqrt(1 - y**(4/3)))



def phi_N(lam, N):

    """
    Args:
        N: size of alphabet

    """
    y = np.zeros_like(lam)
    for k in range(1, N+1):

        a_k = k*(1+k)*np.log2((k+1)/k)
        slc = np.where(((k-1)/k <= lam) & (lam  <= k/(k+1)))
        y[slc] += a_k*(lam[slc] - (k-1)/k) + np.log2(k)
    return y

def edges_of_phi(N):
    xvals = []
    yvals = []
    for k in range(1, N+1):
        x = k / (k+1)
        a_k = k*(1+k)*np.log2((k+1)/k)
        xvals.append(x)
        yvals.append(a_k*(x - (k-1)/k) + np.log2(k))
    return xvals, yvals

def fano_ineq(lam, N):
    # compute the upper bound of entropy as a function of perr
    return binary_entropy(lam) + lam*np.log2(N-1)


def cica_Hmax(p, N, M):
    """Compute the maximum entropy of a _deterministic_ function acting on p
    
    This computes the maximum according to (Cicales, 2017), for a distribuiton
    p on alphabet of size N, with a codomain of size M.
    """
    assert len(p) == N
    # compute the r operator by finding the pivot index
    current = p[0]
    cumsum = p[0]
    pivot = 1
    while current >= (1 - cumsum) / (M - pivot):
        pivot += 1
        current = p[pivot + 1]
        cumsum += current

    out = np.zeros(M)
    out[:pivot] = p[:pivot]
    out[pivot:] = (1 - cumsum) / (M - pivot)
    return out

