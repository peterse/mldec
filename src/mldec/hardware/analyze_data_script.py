import matplotlib.pyplot as plt
from mldec.hardware.topological_codes.postprocessing import reshape_and_verify_correspondence
from mldec.datasets.reps_exp_rep_code_data import build_syndrome_2D, make_exp_dataset_name, load_data

import numpy as np

def make_visualization_by_round(X):
    """Make a histogram of several properties of the syndromes.

    We will return two histograms:
        histogram_by_round: The histogram over all (single-round) syndromes, aggregated
         between different rounds.
         histogram_by_bit: Histogram over all bit-positions in the 2D syndrome, aggregated
         over the n-1 bits in the syndrome.
    
    X has shape (n_data, repetitions, n-1). """

    histogram_by_round = np.zeros(X.shape[2] + 1)
    histogram_by_bit = np.zeros(X.shape[1] + 1)
    out = np.zeros(X.shape[2] + 1)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            histogram_by_round[int(np.sum(X[i, j,:]))] += 1
    for i in range(X.shape[0]):
        for j in range(X.shape[2]):
            histogram_by_bit[int(np.sum(X[i, :, j]))] += 1
    histogram_by_round = histogram_by_round / np.sum(histogram_by_round)
    histogram_by_bit = histogram_by_bit / np.sum(histogram_by_bit)
    return histogram_by_round, histogram_by_bit

delay_factors = [0, 1, 1.1, 1.2, 1.3, 1.4, 2, 3, 4, 5]

ANALYZE_DATA = 2
if ANALYZE_DATA == 1:


    for (n, T) in [(5, 5), (5, 6), (5, 7), (9, 10)]:
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        # load the data
        out = {}
        for delay_factor in delay_factors:
            fname = make_exp_dataset_name(n, T, delay_factor)
            X, y = load_data(fname)
            out[delay_factor] = (X, y)


        for beta, (X, y) in out.items():
            # these are not the syndromes, but rather the syndrome difference per round.
            syndromes = np.array([build_syndrome_2D(X[i,:,:]) for i in range(X.shape[0])])
            syndromes = syndromes.reshape(X.shape)
            histogram_by_round, histogram_by_bit = make_visualization_by_round(syndromes)
            ax[0].plot(range(len(histogram_by_round)), histogram_by_round, label=f"beta={beta}", marker="o")
            ax[1].plot(range(len(histogram_by_bit)), histogram_by_bit, label=f"beta={beta}", marker="o")
            
        ax[0].legend()
        ax[1].legend()
        ax[0].set_xlabel("weight of syndrome")
        ax[1].set_xlabel("weight over all time")
        ax[0].set_ylabel("fraction of syndromes")
        ax[0].semilogy()
        ax[1].semilogy()
        plt.savefig(f"syndrome_weights_n{n}_T{T}.png")

elif ANALYZE_DATA == 2:
    n = 9
    T = 10
    out = {}
    for delay_factor in delay_factors[1:]:
        fname = make_exp_dataset_name(n, T, delay_factor)
        X, y = load_data(fname)
        out[delay_factor] = (X, y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 7))

    for beta, (X, y) in out.items():
        for i in range(3):
            # get a random subset of range(9):
            indices = np.arange(i, i+5)
            X_subset = X[:,:,indices]
            # these are not the syndromes, but rather the syndrome difference per round.
            syndromes = np.array([build_syndrome_2D(X_subset[i,:,:]) for i in range(X_subset.shape[0])])
            syndromes = syndromes.reshape((X.shape[0], X.shape[1], 5))

            histogram_by_round, histogram_by_bit = make_visualization_by_round(syndromes)
            axes[i].plot(range(len(histogram_by_round)), histogram_by_round, label=f"beta={beta}", marker="o")
            axes[i].legend()
            axes[i].semilogy()
            axes[i].set_xlabel("weight of syndrome")
            axes[i].set_ylabel("fraction of syndromes")
            axes[i].set_title(f"subset {indices}")

    plt.savefig(f"syndrome_weights_n{n}_T{T}_subset5.png",  bbox_inches="tight")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    slices = range(4, 9)
    for beta, (X, y) in out.items():
        for i, x in enumerate(slices):
            # get a random subset of range(9):
            X_subset = X[:,:,:x]
            # these are not the syndromes, but rather the syndrome difference per round.
            syndromes = np.array([build_syndrome_2D(X_subset[i,:,:]) for i in range(X_subset.shape[0])])
            syndromes = syndromes.reshape((X.shape[0], X.shape[1], x))

            histogram_by_round, histogram_by_bit = make_visualization_by_round(syndromes)
            axes[i].plot(range(len(histogram_by_round)), histogram_by_round, label=f"beta={beta}", marker="o")
            axes[i].legend()
            axes[i].semilogy()
            axes[i].set_xlabel("weight of syndrome")
            axes[i].set_ylabel("fraction of syndromes")

    plt.savefig(f"syndrome_weights_n{n}_T{T}_subset5678.png", bbox_inches="tight")
