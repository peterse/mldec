import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeTorino

def create_simple_circuits():
    """Create test circuits with two classical registers: c1 and c2."""
    circuits = []
    # Circuit 1
    q = QuantumRegister(2, 'q')
    c1 = ClassicalRegister(1, 'c1')
    c2 = ClassicalRegister(1, 'c2')
    qc1 = QuantumCircuit(q, c1, c2)
    qc1.h(q[0])
    qc1.cx(q[0], q[1])
    qc1.measure(q[0], c1[0])
    qc1.measure(q[1], c2[0])
    circuits.append(qc1)
    # Circuit 2
    q2 = QuantumRegister(2, 'q')
    c1b = ClassicalRegister(1, 'c1')
    c2b = ClassicalRegister(1, 'c2')
    qc2 = QuantumCircuit(q2, c1b, c2b)
    qc2.x(q2[0])
    qc2.measure(q2[0], c1b[0])
    qc2.measure(q2[1], c2b[0])
    circuits.append(qc2)
    return circuits

def analyze_formats():
    """Analyze the differences between AerSimulator.run and Sampler.run outputs."""
    print("=== Format Analysis ===")
    
    # Create test circuits
    circuits = create_simple_circuits()
    shots = 100
    
    # Set up backend and noise model
    device = FakeTorino()
    noise_model = NoiseModel.from_backend(
        device, thermal_relaxation=True, gate_error=True, readout_error=True
    )
    backend = AerSimulator.from_backend(device, noise_model=noise_model)
    
    # Method 1: AerSimulator.run
    print("\n1. AerSimulator.run method:")
    # Transpile circuits to backend basis gates
    from qiskit import transpile
    transpiled_circuits = transpile(circuits, backend)
    aer_job = backend.run(transpiled_circuits, shots=shots)
    aer_result = aer_job.result()
    
    print(f"Type of aer_result: {type(aer_result)}")
    print(f"aer_result attributes: {dir(aer_result)}")
    print(f"aer_result.get_counts(0): {aer_result.get_counts(0)}")
    print(f"aer_result.get_counts(1): {aer_result.get_counts(1)}")
    
    # Method 2: Sampler.run
    print("\n2. Sampler.run method:")
    sampler = Sampler(backend)
    sampler_job = sampler.run(transpiled_circuits, shots=shots)
    sampler_result = sampler_job.result()
    
    print(f"Type of sampler_result: {type(sampler_result)}")
    print(f"sampler_result attributes: {dir(sampler_result)}")
    print(f"Length of sampler_result: {len(sampler_result)}")
    print(f"sampler_result[0]: {sampler_result[0]}")
    print(f"sampler_result[0].data: {sampler_result[0].data}")
    print(f"sampler_result[0].data attributes: {dir(sampler_result[0].data)}")
    
    # Try to access the data
    try:
        print(f"sampler_result[0].data.c1: {sampler_result[0].data.c1}")
        print(f"sampler_result[0].data.c1.get_counts(): {sampler_result[0].data.c1.get_counts()}")
    except AttributeError as e:
        print(f"Error accessing sampler data: {e}")
    
    return aer_result, sampler_result, transpiled_circuits

def convert_aer_to_sampler_format(aer_result, circuits):
    """
    Convert AerSimulator.run result to Sampler.run format, supporting arbitrary classical register names.
    """
    print("\n=== Converting AerSimulator format to Sampler format ===")
    
    # Create a mock SamplerPubResult class to mimic the expected structure
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
                for _ in range(count):
                    arr.append([int(b) for b in bits])
            return np.array(arr).reshape((self.num_shots, self.nbits))
    
    # Get counts for each circuit
    converted_results = []
    for i, circuit in enumerate(circuits):
        counts = aer_result.get_counts(i)
        # Get classical register info
        reg_bins = {}
        cregs = circuit.cregs
        # Qiskit output order: cregs in order of addition
        for reg_index, creg in enumerate(cregs):
            regname = creg.name
            nbits = creg.size
            reg_bins[regname] = MockRegisterBin(regname, counts, reg_index, nbits, sum(counts.values()))
        data_bin = MockDataBin(reg_bins)
        pub_result = MockSamplerPubResult(data_bin)
        converted_results.append(pub_result)
    
    return converted_results

def test_conversion():
    """Test the conversion function with simple circuits."""
    print("=== Testing Conversion ===")
    
    # Analyze formats
    aer_result, sampler_result, circuits = analyze_formats()
    
    # Convert AerSimulator format to Sampler format
    converted_results = convert_aer_to_sampler_format(aer_result, circuits)
    
    print(f"\nConverted results length: {len(converted_results)}")
    for i, result in enumerate(converted_results):
        print(f"Converted result {i}: {result}")
        print(f"  Data: {result.data}")
        for reg in circuits[i].cregs:
            regname = reg.name
            print(f"  Register {regname}: {getattr(result.data, regname)}")
            print(f"    Counts: {getattr(result.data, regname).get_counts()}")
            print(f"    Bool array: {getattr(result.data, regname).to_bool_array()}")
        print(f"  Num shots: {sum(getattr(result.data, reg.name).num_shots for reg in circuits[i].cregs)}")

if __name__ == "__main__":
    test_conversion() 