import numpy as np

def kbits(n, k):
    """Generate integer form for all length-n bitstrings of weight k.

    Output indices are ordered consistently but arbitrarily.

    DISCLAIMER: ripped from StackOverflow, I don't take credit for this code.
    
    Args:
        n, k: integers

    Returns:
        Generator for indices that are ordered by their binary weight.
    """
    limit = 1 << n
    val = (1 << k) - 1
    while val < limit:
        yield val
        minbit = val & -val  #rightmost 1 bit
        fillbit = (val + minbit) & ~val  #rightmost 0 to the left of that bit
        val = val + minbit | (fillbit // (minbit << 1)) - 1


def idxsort_by_weight(n):
    """Construct a list of all length-n bitstrings sorted by weight.

    Within each weight class strings are sorted arbitrariy (based on the
    implementation of `kbits`).

    Args:
        n: Number of bits

    Returns:
        List[Int] with length 2**n containing integers that are sorted by
            binary weight
    """
    out = [0]
    for k in range(1, n + 1):
        out += list(kbits(n, k))
    return np.array(out, dtype=int)


def binarr(m):
    """Produce an ordered column of all binary vectors length m.

    Example for m=3:
        array([[0, 0, 0],
               [0, 0, 1],
               [0, 1, 0],
               [0, 1, 1],
               [1, 0, 0],
               [1, 0, 1],
               [1, 1, 0],
               [1, 1, 1]])
    """
    d = idxsort_by_weight(m)
    return (((d[:, None] & (1 << np.arange(m, dtype=int)))) > 0).astype(int)[:, ::-1]


def bits_to_ints(bits):
    # convert a boolean/integer array of ints to its integer value 
    m, n = bits.shape # each row is abinary string
    a = 2**np.arange(n)[::-1]
    nums = bits @ a # integer
    return nums


def ints_to_bits(nums, n):
    # convert an integer array to its binary repr
    return (((nums[:,None] & (1 << np.arange(n)))) > 0).astype(bool)