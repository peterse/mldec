import numpy as np
from mldec.utils.bit_tools import binarr
import math
from mldec.datasets import toy_problem_data
import matplotlib.pyplot as plt

def compute_pr_unimportant_examples(n, p1, p2):
    # unimportant means good but not important
    # this is just for verification, all unimportant + important + bad examples should sum to 1
    all_bitstrings = binarr(n)
    config = {'p1': p1, 'p2': p2}
    probs = toy_problem_data.noise_model(all_bitstrings, n, config, numpy=True)
    idx_sorted = np.argsort(probs)
    halfidx = 2**(n-1)

    out = probs[idx_sorted][halfidx:][::-1]
    sorted_all = all_bitstrings[idx_sorted][halfidx:][::-1]
    # only keep sorted_all that have less than n//2 set bits
    idx_keep = np.where(sorted_all.sum(axis=1) < n//2)
    kept = sorted_all[idx_keep]
    kept_probs = out[idx_keep]
    return kept_probs.sum()


def compute_pr_important_examples(n, p1, p2):
    # all important examples have exactly n//2 set bits IN THE WELL-ORDERED SETTING
    assert n % 2 == 0
    tot = 0
    half_n = n//2
    # This loop _just_ counts errors with exactly n//2 set bits
    for i in range(0, n//4 + 1): # i is the set bits on the LHS, this will only go up to n//4
        temp = math.comb(half_n, i) ** 2 * p2**i * (1-p2)**(half_n - i) * p1**(half_n-i) * (1-p1)**i
        # i is the number of set bits in the first half (prob=p1), and there are n//2-i set bits in the second half (prob=p2)
        temp = math.comb(half_n, i) ** 2 * p2**i * (1-p2)**(half_n - i) * p1**(half_n-i) * (1-p1)**i
        if i == n//4 and n % 4 == 0:
            temp /= 2
        tot += temp
    # the final probability 
    return tot/2


def compute_pr_bad_examples(n, p1, p2):
    assert n % 2 == 0
    tot = 0
    half_n = n//2
    # start by computing the weight n//2 bad examples: This loop _just_ counts errors with exactly n//2 set bits
    for i in range(0, n//4 + 1): # i is the set bits on the LHS, this will only go up to n//4
        temp = math.comb(half_n, i) ** 2 * p1**i * (1-p1)**(half_n - i) * p2**(half_n-i) * (1-p2)**i
        # i is the number of set bits in the first half (prob=p1), and there are n//2-i set bits in the second half (prob=p2)
        temp = math.comb(half_n, i) ** 2 * p1**i * (1-p1)**(half_n - i) * p2**(half_n-i) * (1-p2)**i
        if i == n//4 and n % 4 == 0:
            temp /= 2
        tot += temp
    
    # then, add up all weight > n//2 examples
    # all of these are fair game
    for w in range(n//2+1, n+1):
        # compute all pairs of integers that sum to w, with neither greater than n//2
        for i in range(max(0, w - n//2), min(w, n//2)+1):
            # again, i is the set bits on the LHS, so there are w-i set bits on the RHS
            pr = math.comb(n//2, i) * math.comb(n//2, w-i) * p1**i * (1-p1)**(n//2 - i) * p2**(w-i) * (1-p2)**(n//2 - (w-i))
            tot += pr
    return tot


def compute_pr_bad_examples_v2(n, p1, p2):
    # this is the brute force method and is slow
    all_bitstrings = binarr(n)
    config = {'p1': p1, 'p2': p2}
    probs = toy_problem_data.noise_model(all_bitstrings, n, config, numpy=True)
    idx_sorted = np.argsort(probs)
    halfidx = 2**(n-1)
    return probs[idx_sorted][:halfidx].sum()


def compute_pr_important_bad(n, p, alpha):
    # this is the brute force method and is slow
    all_bitstrings = binarr(n)
    config = {'p': p, 'alpha': alpha}
    probs = toy_problem_data.noise_model(all_bitstrings, n, config, numpy=True)
    idx_sorted = (-probs).argsort()
    probs_sorted = probs[idx_sorted]
    bitstrings_sorted = all_bitstrings[idx_sorted]

    # The bad examples are, by definition, the least likely half.
    # this only works for the k=1 rep code.
    halfidx = 2**(n-1)
    pr_bad = probs_sorted[halfidx:].sum()
    pr_imp = 0
    ct = 1
    well_ordered = True
    for (e, p) in zip(bitstrings_sorted, probs_sorted):
        if ct > halfidx:
            break
        wt_e = e.sum()
        if wt_e == n//2:
            pr_imp += p/2
        elif wt_e > ( n - wt_e):
            # print("error: ", e, p, 1 - e)
            pr_imp += p
            well_ordered = False
        ct += 1
    return pr_imp, pr_bad, well_ordered


def histogram_good_important_bad(n, p, alpha):
    assert n % 2 == 0
    # this is the brute force method and is slow
    all_bitstrings = binarr(n)
    config = {'p': p, 'alpha': alpha}
    probs = toy_problem_data.noise_model(all_bitstrings, n, config, numpy=True)
    idx_sorted = (-probs).argsort()
    probs_sorted = probs[idx_sorted]
    bitstrings_sorted = all_bitstrings[idx_sorted]

    bad = []
    # The bad examples are, by definition, the least likely half.
    # this only works for the k=1 rep code.
    halfidx = 2**(n-1)
    bad = bitstrings_sorted[halfidx:]
    bad_wts = bad.sum(axis=1)
    pr_bad = probs_sorted[halfidx:]
    bad_wt_hist = {i: 0 for i in range(n+1)}
    for (wt, p) in zip(bad_wts, pr_bad):
        bad_wt_hist[wt] += p

    good_wt_hist = {i: 0 for i in range(n+1)}
    good = bitstrings_sorted[:halfidx]
    good_wts = good.sum(axis=1)
    pr_good = probs_sorted[:halfidx]
    for (wt, p) in zip(good_wts, pr_good):
        good_wt_hist[wt] += p

    imp_wt_hist = {i: 0 for i in range(n+1)}
    ct = 1
    well_ordered = True
    for (e, p) in zip(bitstrings_sorted, probs_sorted):
        if ct > halfidx:
            break
        wt_e = e.sum()
        if wt_e == n//2:
            imp_wt_hist[wt_e] += p/2
        elif wt_e > ( n - wt_e):
            imp_wt_hist[wt_e] += p
        ct += 1

    all_wt_hist = {i: 0 for i in range(n+1)}
    for (e, p) in zip(bitstrings_sorted, probs_sorted):
        wt_e = e.sum()
        all_wt_hist[wt_e] += p

    unimp_wt_hist = {i: 0 for i in range(n+1)}
    for i in range(n+1):
        unimp_wt_hist[i] = good_wt_hist[i] - imp_wt_hist[i]

    return good_wt_hist, unimp_wt_hist, imp_wt_hist, bad_wt_hist, all_wt_hist