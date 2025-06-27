from mldec.datasets.reps_toric_code_data import make_sampler
from mldec.datasets import reps_toric_code_data
from mldec.models import baselines
from mldec.utils import evaluation
import torch
device = torch.device("cpu")
import tqdm
import multiprocessing
import pandas as pd


# Baseline accuracies: set up pymatching decoder for validation set; do this directly on stim detection events
base_config = {
    "repetitions": 5,
    "code_size": 3,
    "p": 0.001,
    "beta": 1.0,
}
# create a small validation dataset
n_test = int(1e7)

def func(beta_seed, base_config=base_config, n_test=n_test):
    """Initialize MWPM with beta*p, evaluate on errors sampled from p."""
    beta, seed = beta_seed
    data_val, triv_val, stim_data_val, observable_flips_val = reps_toric_code_data.sample_dataset(n_test, base_config, device, seed=seed)
    training_config = base_config.copy()
    training_config["beta"] = beta
    _, _, detector_error_model = reps_toric_code_data.make_sampler(training_config)
    mwpm_decoder = baselines.CyclesMinimumWeightPerfectMatching(detector_error_model)
    minimum_weight_correct_nontrivial = evaluation.evaluate_mwpm(stim_data_val, observable_flips_val, mwpm_decoder).item()
    minimum_weight_val_acc = (minimum_weight_correct_nontrivial + triv_val) / n_test
    return minimum_weight_val_acc

def process_beta(beta_seed):
    return func(beta_seed, base_config=base_config, n_test=n_test)


if __name__ == "__main__":
    # generate a list of (beta, seed) for beta in range(1, 15) and seed in range(10)
    beta_list = range(1, 20)
    ntrials = 30
    beta_seed_list = [(beta, 1234+ seed) for beta in beta_list for seed in range(ntrials)]
    print("mapping over", len(beta_seed_list), "beta/seeds")

    pool = multiprocessing.Pool(processes=20)
    mapped_values = list(tqdm.tqdm(pool.imap_unordered(process_beta, beta_seed_list), total=len(beta_seed_list)))
    beta_seed_acc_list = []
    for (beta, seed), acc in zip(beta_seed_list, mapped_values):
        beta_seed_acc_list.append((beta, seed, acc, n_test))
    print(beta_seed_acc_list)
    
    # Convert the list of tuples to a DataFrame
    df = pd.DataFrame(beta_seed_acc_list, columns=['beta', 'seed', 'accuracy', 'ntest'])
    
    # Sort by beta and seed for better readability
    df = df.sort_values(['beta', 'seed'])
    
    # Display the DataFrame
    print("\nMWPM Benchmark Results:")
    print(df)

    # Save the DataFrame to a CSV file
    df.to_csv('mwpm_benchmark_results.csv', index=False)
    