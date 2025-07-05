import numpy as np


def reshape_and_verify_correspondence(X, y):
    """
    Reshape X and y while preserving one-to-one correspondence.
    
    Args:
        X: shape (n_trials, repetitions, n_data, n-1)
        y: shape (n_trials, n_data)
    
    Returns:
        X_reshaped: shape (n_trials * n_data, repetitions, n-1)
        y_reshaped: shape (n_trials * n_data)
    """
    n_trials, repetitions, n_data, n_minus_1 = X.shape
    
    # Reshape X: (n_trials, repetitions, n_data, n-1) -> (n_trials * n_data, repetitions, n-1)
    X_reshaped = X.transpose(0, 2, 1, 3)  # (n_trials, n_data, repetitions, n-1)
    X_reshaped = X_reshaped.reshape(-1, repetitions, n_minus_1)  # (n_trials * n_data, repetitions, n-1)
    
    # Reshape y: (n_trials, n_data) -> (n_trials * n_data)
    y_reshaped = y.reshape(-1)  # (n_trials * n_data)
    
    return X_reshaped, y_reshaped

def verify_correspondence(X_original, y_original, X_reshaped, y_reshaped):
    """
    Verify that the correspondence between X and y is preserved after reshaping.
    
    Args:
        X_original: shape (n_trials, repetitions, n_data, n-1)
        y_original: shape (n_trials, n_data)
        X_reshaped: shape (n_trials * n_data, repetitions, n-1)
        y_reshaped: shape (n_trials * n_data)
    """
    n_trials, repetitions, n_data, n_minus_1 = X_original.shape
    
    print(f"Original shapes: X={X_original.shape}, y={y_original.shape}")
    print(f"Reshaped shapes: X={X_reshaped.shape}, y={y_reshaped.shape}")
    print()
    
    # Test correspondence for a few specific indices
    test_indices = [
        (0, 0),  # trial 0, data 0
        (0, 1),  # trial 0, data 1
        (1, 0),  # trial 1, data 0
        (n_trials-1, n_data-1)  # last trial, last data
    ]
    
    print("Verifying correspondence:")
    for trial_idx, data_idx in test_indices:
        # Original indexing
        original_y = y_original[trial_idx, data_idx]
        original_X = X_original[trial_idx, :, data_idx, :]  # (repetitions, n-1)
        
        # New indexing: trial_idx * n_data + data_idx
        new_idx = trial_idx * n_data + data_idx
        new_y = y_reshaped[new_idx]
        new_X = X_reshaped[new_idx, :, :]  # (repetitions, n-1)
        
        # Check if they match
        y_match = np.allclose(original_y, new_y)
        X_match = np.allclose(original_X, new_X)
        
        print(f"  Trial {trial_idx}, Data {data_idx} (new_idx={new_idx}):")
        print(f"    y match: {y_match} (original={original_y}, new={new_y})")
        print(f"    X match: {X_match}")
        print()
    
    # Verify the mapping formula works for all indices
    print("Verifying mapping formula for all indices...")
    all_match = True
    for trial_idx in range(n_trials):
        for data_idx in range(n_data):
            new_idx = trial_idx * n_data + data_idx
            if not np.allclose(y_original[trial_idx, data_idx], y_reshaped[new_idx]):
                print(f"Mismatch at trial {trial_idx}, data {data_idx}")
                all_match = False
                break
        if not all_match:
            break
    
    if all_match:
        print("✓ All correspondences verified successfully!")
    else:
        print("✗ Correspondence verification failed!")
    
    return all_match

### THe following code converts AerSimulator job result into sampler job result format.
class MockSamplerPubResult:
    def __init__(self, data, metadata=None):
        self.data = data
        self.metadata = metadata or {}

class MockDataBin:
    def __init__(self, reg_bins):
        for regname, binobj in reg_bins.items():
            setattr(self, regname, binobj)

class MockRegisterBin:
    def __init__(self, regname, counts_dict, reg_index, nbits, num_shots):
        self.regname = regname
        self.counts_dict = counts_dict
        self.reg_index = reg_index
        self.nbits = nbits
        self.num_shots = num_shots
        
    def get_counts(self):
        return self.counts_dict
        
    def to_bool_array(self):
        # Reconstruct bitstrings for this register from the counts
        arr = []
        for bitstring, count in self.counts_dict.items():
            # bitstring is a full output string, e.g. '0 1' or '01' or '0 1 0'
            # We need to extract the bits for this register
            # For Qiskit, if multiple classical registers, bitstring is space-separated in order of cregs
            # We'll split and take the reg_index-th part
            if ' ' in bitstring:
                bits = bitstring.split(' ')[self.reg_index]
            else:
                # If only one register, or Qiskit output is not space-separated
                bits = bitstring
            # Each bit in bits (for multi-bit registers)
            arr.extend([[int(b) for b in bits]] * count)
        result = np.array(arr).reshape((self.num_shots, self.nbits))
        return result

def convert_aer_to_sampler_format(aer_result, circuits):
    """
    Convert AerSimulator.run result to Sampler.run format, supporting arbitrary classical register names.
    """
    converted_results = []
    for i, circuit in enumerate(circuits):
        counts = aer_result.get_counts(i)
        # qiskit uses a endianness that doesn't match their indexing, so we reverse the bitstrings.
        reversed_counts = {k[::-1]: v for k, v in counts.items()}

        # Get classical register info
        reg_bins = {}
        cregs = circuit.cregs
        # Qiskit output order: cregs in order of addition
        for reg_index, creg in enumerate(cregs):
            regname = creg.name
            nbits = creg.size
            reg_bins[regname] = MockRegisterBin(regname, reversed_counts, reg_index, nbits, sum(counts.values()))
        data_bin = MockDataBin(reg_bins)
        pub_result = MockSamplerPubResult(data_bin)
        converted_results.append(pub_result)
    return converted_results 