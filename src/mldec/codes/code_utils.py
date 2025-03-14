import stim

from itertools import chain, combinations
from functools import reduce


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def operator_sequence_to_stim(lst, n):
    if len(lst) > 0:
        out = reduce(lambda P, Q: P*Q, lst)
    else:
        out = stim.PauliString("_" * n) 
    return out