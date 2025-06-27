from qiskit.circuit import Delay, Instruction, QuantumCircuit
from typing import Union
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import IGate
from qiskit.converters import circuit_to_dag, dag_to_circuit

def insert_idle_delays(
    circuit: QuantumCircuit,
    delay_factor: Union[int, float],
    backend
) -> QuantumCircuit:
    """
    Return a new circuit that is the same as `circuit` but with a Delay inserted
    after every non-Delay instruction on the same qubits.

    Args:
        circuit:       The original QuantumCircuit.
        delay_factor:  This is the _total_ factor by which the timing of the gate is increased. For example, if
                       delay_factor=1, we do not add any additional time. if delay_factor=2, we add a delay equal to
                       the duration of each (non-virtual) gate.
        backend:       The backend to use for the durations.

    Returns:
        QuantumCircuit: A new circuit with delays interleaved.
    """
    assert delay_factor >= 1, "delay_factor must be at least 1"
    # Recreate circuit skeleton (same qregs & cregs)
    new_circ = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name + '_idle')
    
    # skip if there's no delay, so we don't have to look at wait gates

    if delay_factor == 1:
        return circuit
    
    durations = backend.target.durations()
    for instr, qargs, cargs in circuit.data:
        # 1) If it's not already a Delay, insert an idle before the gate
        if isinstance(instr, Delay):
            new_circ.append(instr, qargs, cargs)
            continue
        if qargs:
            qubits = [q._index for q in qargs]
            instr_name = instr.name
            print(instr_name, instr)
            instr_duration = durations.get(instr_name, qubits, unit='dt')
            if instr_duration > 0:
                # Delay by a fraction of the instruction duration
                delay_time = instr_duration * (delay_factor - 1)
                new_circ.delay(delay_time, qargs, unit='dt')
        # 2) Copy the original instruction
        new_circ.append(instr, qargs, cargs)
    return new_circ



def insert_idle_delays_dag(circuit, delay_factor, backend):
    dag = circuit_to_dag(circuit)
    durations = backend.target.durations()

    # 1. Make a fresh, empty copy (with same qubits & clbits).
    new_dag = dag.copy_empty_like()
    
    # 2. Walk through all operation nodes in topo order
    for node in dag.topological_op_nodes():
        # 3. Insert an identity (idle) on the same wires, conditioned on whether the operation has duration
        if node.qargs and delay_factor > 1:
            qubits = [q._index for q in node.qargs]
            instr_name = node.op.name
            instr_duration = durations.get(instr_name, qubits, unit='dt')
            if instr_duration > 0:
                # Delay by a fraction of the instruction duration
                delay_time = instr_duration * (delay_factor - 1)
                for qubit in node.qargs:
                    new_dag.apply_operation_back(Delay(delay_time), [qubit], [])

        new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

    new_circuit = dag_to_circuit(new_dag)
    return new_circuit
