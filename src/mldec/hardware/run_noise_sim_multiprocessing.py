from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.transpiler import Layout
from qiskit import QuantumCircuit, transpile
from mldec.hardware.topological_codes.circuits import RepetitionCode, HardwarePhaseFlipRepetitionCode, generate_initial_states, get_target_qubits
from mldec.utils.experiments_ibm import process_jobs
import time

from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from mldec.datasets.reps_exp_rep_code_data import make_exp_dataset_name, save_data
from mldec.hardware.topological_codes.postprocessing import reshape_and_verify_correspondence
from mldec.hardware.topological_codes.postprocessing import convert_aer_to_sampler_format
from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np



def main():
    #########################################################################################
    # run parameters
    # DON'T FORGET TO MODIFY YOUR ENV FIRST
    # export OMP_NUM_THREADS=72

    max_workers = 72 # num threads

    print("available cpus: ", os.cpu_count(), "using ", max_workers)

    # Set the number of qubits and rounds of syndrome measurement.

    num_train = 4096 # training examples PER TRIAL, total training data is num_train * num_trials
    num_val = 4096 # this is number of validation runs PERTRIAL

    num_trials = 8 # every trial consists of a different initial state.
    delay_factors = [1, 1.1, 1.2, 1.3, 1.4, 2, 3, 4, 5]
    for (n, T) in [(6, 6), (7, 7), (8, 8)]:
        print("=== Simulation Parameters Summary ===")
        print(f"Max workers (threads): {max_workers}")
        print(f"Number of qubits (n): {n}")
        print(f"Syndrome measurement rounds (T): {T}")
        print(f"Training examples per trial: {num_train}")
        print(f"Validation examples per trial: {num_val}")
        print(f"Number of trials: {num_trials}")
        print(f"Delay factors: {delay_factors}")
        print(f"Total training circuits: {num_trials * len(delay_factors)}")
        print(f"Total validation circuits: {num_trials}")
        print(f"Total training shots per delay factor: {num_train * num_trials}")
        print(f"Total validation shots: {num_val * num_trials}")
        print("=====================================")
        print(">>>>> did you remember to export OMP_NUM_THREADS= in the terminal? <<<<<")
        #########################################################################################

        target_qubits = get_target_qubits(n)
        device = FakeTorino()
        noise_model = NoiseModel.from_backend(
            device, thermal_relaxation=True, gate_error=True, readout_error=True
        )
        backend = AerSimulator.from_backend(device, noise_model=noise_model, method='statevector')


        # We will specify different initial qubit states, but due to bandwidth issues we run many
        # shots per choice of initial state (same as Google 2023 paper).
        train_initial_states, val_initial_states = generate_initial_states(n, num_trials)

        val_initial_states = train_initial_states
        print("WARNING: using the same initial states for training and validation")
        # build the list of circuits to execute
        training_circuits = []
        train_num_shots = []
        tot_train = 0
        for trial in range(num_trials):
            initial_state = train_initial_states[trial]
            for delay_factor in delay_factors:
                # initialize a new 
                code = HardwarePhaseFlipRepetitionCode(n, T, initial_state=initial_state, backend=backend, delay_factor=delay_factor, target_qubits=target_qubits)
                training_circuits.append(code.circuit)
                train_num_shots.append(num_train)
                tot_train += 1

        validation_circuits = []
        val_num_shots = []
        for trial in range(num_trials):
            initial_state = val_initial_states[trial]
            code = HardwarePhaseFlipRepetitionCode(n, T, initial_state=initial_state, backend=backend, delay_factor=1, target_qubits=target_qubits)
            validation_circuits.append(code.circuit)
            val_num_shots.append(num_val)


        #########################################################################################
        # Running the jobs

        print(f"submitting {tot_train} training circuits and {len(validation_circuits)} validation circuits")
        exc = ProcessPoolExecutor(max_workers=max_workers)
        # backend.set_options(executor=exc)
        backend.set_options(max_parallel_experiments =max_workers)
        backend.set_options(max_parallel_threads =max_workers)
        # Transpile circuits to backend basis gates
        transpiled_training_circuits = transpile(training_circuits, backend)
        transpiled_validation_circuits = transpile(validation_circuits, backend)
        
        start_time = time.time()
        training_job = backend.run(transpiled_training_circuits, shots=num_train)
        validation_job = backend.run(transpiled_validation_circuits, shots=num_val)
        train_job_result = training_job.result()
        val_job_result = validation_job.result()
        end_time = time.time()
        print(f"Sim time taken: {end_time - start_time} seconds")

        # CONVERT THE TRAINING JOB INTO THE FORMAT OF sampler.Run
        print("Converting AerSimulator results to Sampler format...")
        converted_train_result = convert_aer_to_sampler_format(train_job_result, transpiled_training_circuits)
        converted_val_result = convert_aer_to_sampler_format(val_job_result, transpiled_validation_circuits)

        # POSTPROCESSING
        # Convert the nested lists of jobs into a dictionary of the form {beta: (X, Y)}
        # this also processes the data into DetectorDDD format
        out = process_jobs(n, T, converted_train_result, converted_val_result, delay_factors, train_initial_states, val_initial_states)


        for beta, (X, y) in out.items():
            fname = make_exp_dataset_name(n, T, beta)
            # current shape: X: (n_trials, repetitions, n_data, n-1), y: (n_trials, n_data)
            X, y = reshape_and_verify_correspondence(X, y)
            # new shapes: (n_trials*n_data, repetitions, n-1), y: (n_trials*n_data)

            save_data(X, y, fname)


if __name__ == "__main__":
    main()

    






    