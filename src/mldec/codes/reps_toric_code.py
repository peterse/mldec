import pymatching
import numpy as np
from mldec.datasets import reps_toric_code_data
from mldec.models import baselines
from mldec.utils import evaluation

def estimate_mwpm_error(dataset_config, n_test):
    sampler, detector_coordinates, detector_error_model = reps_toric_code_data.make_sampler(dataset_config)
    n_test = int(n_test)

    # sample detection events and observable flips
    stim_data, observable_flips = sampler.sample(shots=n_test, separate_observables=True)
    non_empty_indices = (np.sum(stim_data, axis = 1) != 0)
    triv_val = len(observable_flips[~ non_empty_indices])
    stim_data_val = stim_data[non_empty_indices, :]
    observable_flips_val = observable_flips[non_empty_indices]

    # code for verification purposes only:
    # # Configure a decoder using the circuit.
    # # detector_error_model = circuit.detector_error_model(decompose_errors=True)
    # matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # # Run the decoder.
    # predictions = matcher.decode_batch(stim_data_val)

    # # Count the mistakes.
    # num_errors = 0
    # for shot in range(len(stim_data_val)):
    #     actual_for_shot = observable_flips_val[shot]
    #     predicted_for_shot = predictions[shot]
    #     if not np.array_equal(actual_for_shot, predicted_for_shot):
    #         num_errors += 1

    mwpm_decoder = baselines.CyclesMinimumWeightPerfectMatching(detector_error_model)
    minimum_weight_correct = evaluation.evaluate_mwpm(stim_data_val, observable_flips_val, mwpm_decoder)
    return (1 - (minimum_weight_correct + triv_val) / n_test).item()
