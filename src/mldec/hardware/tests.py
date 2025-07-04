import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.transpiler import Layout
from qiskit import QuantumCircuit, transpile
from mldec.hardware.topological_codes.circuits import RepetitionCode, HardwarePhaseFlipRepetitionCode, generate_initial_states, get_target_qubits
from mldec.utils.experiments_ibm import process_jobs
from mldec.hardware.postprocessing import convert_aer_to_sampler_format

from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from concurrent.futures import ThreadPoolExecutor

def test_real_circuit_conversion():
    """Test the conversion with real repetition code circuits."""
    print("=== Testing Real Circuit Conversion ===")
    
    #########################################################################################
    # run parameters
    max_workers = 2  # num threads
    max_job_size = 1  # number of circuits per worker

    # Set the number of qubits and rounds of syndrome measurement.
    n = 5
    T = 5

    num_train = 10  # training examples PER TRIAL, total training data is num_train * num_trials
    num_val = 10    # this is number of validation runs PER INITIAL STATE TRIAL

    num_trials = 2
    delay_factors = [1, 2]

    #########################################################################################

    target_qubits = get_target_qubits(n)
    device = FakeTorino()
    noise_model = NoiseModel.from_backend(
        device, thermal_relaxation=True, gate_error=True, readout_error=True
    )
    backend = AerSimulator.from_backend(device, noise_model=noise_model)

    # We will specify different initial qubit states, but due to bandwidth issues we run many
    # shots per choice of initial state (same as Google 2023 paper).
    train_initial_states, val_initial_states = generate_initial_states(n, num_trials, seed=1234)
    print("train initial states: ", train_initial_states)
    print("val initial states: ", val_initial_states)

    # build the list of circuits to execute
    training_circuits = []
    train_num_shots = []
    tot_train = 0
    for trial in range(num_trials):
        initial_state = train_initial_states[trial]
        print("initial state: ", initial_state)
        for delay_factor in delay_factors:
            # initialize a new 
            code = HardwarePhaseFlipRepetitionCode(n, T, initial_state=initial_state, backend=backend, delay_factor=delay_factor, target_qubits=target_qubits)
            training_circuits.append(code.circuit)
            train_num_shots.append(num_train)
            tot_train += 1

    validation_circuits = []
    val_num_shots = []
    for trial in range(num_trials):
        print("val initial state: ", val_initial_states[trial])
        initial_state = val_initial_states[trial]
        code = HardwarePhaseFlipRepetitionCode(n, T, initial_state=initial_state, backend=backend, delay_factor=1, target_qubits=target_qubits)
        validation_circuits.append(code.circuit)
        val_num_shots.append(num_val)

    #########################################################################################
    # Running the jobs

    print(f"submitting {tot_train} training circuits and {len(validation_circuits)} validation circuits")
    exc = ThreadPoolExecutor(max_workers=max_workers)
    backend.set_options(executor=exc)
    backend.set_options(max_job_size=max_job_size)
    
    # Transpile circuits to backend basis gates
    transpiled_training_circuits = transpile(training_circuits, backend)
    transpiled_validation_circuits = transpile(validation_circuits, backend)
    
    training_job = backend.run(transpiled_training_circuits, shots=num_train)
    validation_job = backend.run(transpiled_validation_circuits, shots=num_val)
    train_job_result = training_job.result()
    val_job_result = validation_job.result()

    # CONVERT THE TRAINING JOB INTO THE FORMAT OF sampler.Run
    print("Converting AerSimulator results to Sampler format...")
    converted_train_result = convert_aer_to_sampler_format(train_job_result, transpiled_training_circuits)
    converted_val_result = convert_aer_to_sampler_format(val_job_result, transpiled_validation_circuits)

    # POSTPROCESSING
    # Convert the nested lists of jobs into a dictionary of the form {beta: (X, Y)}
    # this also processes the data into DetectorDDD format
    print("Running process_jobs with converted results...")
    try:
        out = process_jobs(n, T, converted_train_result, converted_val_result, delay_factors, train_initial_states, val_initial_states)
        print("SUCCESS: process_jobs completed without errors!")
        
        # Print some basic info about the output
        print(f"Output keys: {list(out.keys())}")
        for delay_factor, (X, Y) in out.items():
            print(f"Delay factor {delay_factor}: X shape {X.shape}, Y shape {Y.shape}")
            
    except Exception as e:
        print(f"ERROR: process_jobs failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_real_circuit_conversion()
    if success:
        print("\n✅ Test PASSED: AerSimulator results successfully converted and processed!")
    else:
        print("\n❌ Test FAILED: Error occurred during processing.") 