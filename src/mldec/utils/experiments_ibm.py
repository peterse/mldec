import numpy as np
from mldec.datasets.toy_problem_data import repetition_pcm


def process_jobs(n, T, train_job_result, val_job_result, delay_factors, train_initial_states, val_initial_states):
    """
    The train_jobs are expected to have been a nested loop of the form:
    for i in range(num_trials):
        for j in range(len(delay_factors)):
            ...
    so that indexing the (i-th trial, j-th delay factor) is accomplished with 
    i*len(delay_factors) + j.

    Process the results of both the traing and validation jobs together. X data will
    have the shape(num_trials, T, shots, n-1) corresponding to 
        (initial state, round, shots, syndrome bits)
    
    Args: 

    Returns:
        Dictionary {delay_factor: (X_tr, Y_tr)}, and delay_factor=0 indicates validation set.
        Each X_tr has shape (num_trials, T, shots, n-1) corresponding to 
            (initial state, round, shots, syndrome bits)
        Each Y_tr has shape (num_trials, shots, 1) corresponding to 
            (initial state, shots, I{initial_state != final_state})
    """

    out = {}
    tr_shots = train_job_result[0].data.data_bit_0.num_shots
    val_shots = val_job_result[0].data.data_bit_0.num_shots
    
    num_trials = len(train_initial_states)
    num_delay_factors = len(delay_factors)
    for delay_factor in delay_factors:
        X_tr = np.zeros((num_trials, T, tr_shots, n - 1))
        Y_tr = np.zeros((num_trials, tr_shots))
        out[delay_factor] = [X_tr, Y_tr]

    # use a dummy value '0' to indicate validation data
    X_val = np.zeros((num_trials, T, val_shots, n - 1))
    Y_val = np.zeros((num_trials, val_shots))
    out[0] = [X_val, Y_val]

    H = repetition_pcm(n)
    for i in range(num_trials):
        train_initial_state = train_initial_states[i][::-1] #endianess
        # get the baseline syndrome: We are going to store 
        # deviation from this expected sydrome as the 'actual' syndrome
        baseline_syndrome = train_initial_state @ H.T % 2
        for j, delay_factor in enumerate(delay_factors):
            train_job_trial_i_delay_j = train_job_result[i*num_delay_factors + j]
            for k in range(T): 
                train_arr = getattr(train_job_trial_i_delay_j.data, f"round_{k}_link_bit_0").to_bool_array() 
                out[delay_factor][0][i, k, :, :] = train_arr.astype(int)
            # now we mod out the baseline induced by the initial state to see a nontrivial syndrome only
            # this is _different_ than forming a detector as flips between rounds.
            out[delay_factor][0][i, :, :, :] = np.logical_xor(out[delay_factor][0][i, :, :, :].astype(int), baseline_syndrome).astype(int)
            
            # compute y values as whether the final state is the same as the initial state,
            # TODO: I guess any individual bitflip counts as a logical observable flip in the conjugate basis.
            final_tr_arr = train_job_trial_i_delay_j.data.data_bit_0.to_bool_array().astype(int)
            Y_tr = final_tr_arr.astype(int)
            tr_errors = ((Y_tr != train_initial_state).astype(int).sum(axis=1) != 0).astype(int)
            out[delay_factor][1][i, :] = tr_errors
        
        # repeat the process for the validation data, which might have different initial states and shapes
        val_initial_state = val_initial_states[i][::-1]  # endianess 
        val_baseline_syndrome = val_initial_state @ H.T % 2
        for k in range(T):
            val_arr = getattr(val_job_result[i].data, f"round_{k}_link_bit_0").to_bool_array() 
            out[0][0][i, k, :, :] = val_arr.astype(int)
        # mod out the basline validation syndrome
        out[0][0][i, :, :, :] = np.logical_xor(out[0][0][i, :, :, :].astype(int), val_baseline_syndrome).astype(int)
        final_val_arr = val_job_result[i].data.data_bit_0.to_bool_array()
        Y_val = final_val_arr.astype(int)
        val_errors = ((Y_val != val_initial_state).astype(int).sum(axis=1) != 0).astype(int)
        out[0][1][i, :] = val_errors
    
    return out