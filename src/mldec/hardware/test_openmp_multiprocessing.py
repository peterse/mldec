import time
import qiskit
import qiskit_aer as qk_aer

parallel_sim = qk_aer.AerSimulator(max_parallel_threads=0, max_parallel_experiments=0, max_parallel_shots=0)

def get_circ():
    circ = qiskit.transpile(qiskit.circuit.library.QuantumVolume(num_qubits=20, depth=5), basis_gates=['u','cx'])
    circ.measure_all()
    return circ

circs6 = [get_circ() for i in range(6)]
circs12 = [get_circ() for i in range(12)]
circs12_2 = [get_circ() for i in range(12)]
circs12_3 = [get_circ() for i in range(12)]

t1 = time.time()
t = parallel_sim.run(circs12).result()
t2 = time.time()
print(t2-t1)

t1 = time.time()
t = parallel_sim.run(circs12_2).result()
t2 = time.time()
print(t2-t1)

t1 = time.time()
t = parallel_sim.run(circs6).result()
t2 = time.time()
print(t2-t1)

t1 = time.time()
t = parallel_sim.run(circs12).result()
t2 = time.time()
print(t2-t1)

t1 = time.time()
t = parallel_sim.run(circs12_2).result()
t2 = time.time()
print(t2-t1)

t1 = time.time()
t = parallel_sim.run(circs12_3).result()
t2 = time.time()
print(t2-t1)