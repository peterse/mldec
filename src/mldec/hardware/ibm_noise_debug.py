# A T1 simulator
from qiskit_ibm_runtime.fake_provider import FakePerth
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_experiments.library import T2Ramsey
import numpy as np

# Create a pure relaxation noise model for AerSimulator
noise_model = NoiseModel.from_backend(
    FakePerth(), thermal_relaxation=True, gate_error=False, readout_error=False
)

# Create a fake backend simulator
backend = AerSimulator.from_backend(FakePerth(), noise_model=noise_model)

qubit = 0
# set the desired delays
delays = list(np.arange(1e-6, 50e-6, 2e-6))

# Create a T2Ramsey experiment. Print the first circuit as an example
exp1 = T2Ramsey((qubit,), delays, osc_freq=1e5)

# Set scheduling method so circuit is scheduled for delay noise simulation
exp1.set_transpile_options(scheduling_method='asap')

# Run experiment
import pdb; pdb.set_trace()
expdata1 = exp1.run(backend=backend, shots=2000, seed_simulator=101)
expdata1.block_for_results()  # Wait for job/analysis to finish.

# Display the figure
# display(expdata1.figure(0))