
import numpy as np
from scipy.stats import unitary_group
import pickle 
import gzip
import io as io_paths

def random_state(n_qubits):
    return unitary_group.rvs(2**n_qubits)[:, 0]


def get_subspace(n_qubits, n_basis, n_samples):
    
    output_states = []
    subspace_basis = (unitary_group.rvs(2**n_qubits)[:, :n_basis]).T
    for _ in range(n_samples):
        c = np.random.rand(n_basis) - 0.5
        linear_combination = 0.j
        for i in range(n_basis):
            linear_combination += c[i] * subspace_basis[i]
        output_states.append(linear_combination / np.linalg.norm(linear_combination))
    return output_states
    
def projection(a, b):
    return np.abs(np.dot(np.conj(a), b))**2


def create_data(n_qubit, measurement_n1, measurement_n2, n_sample, file_name=None, incomplete_tomography=[False, False]):

    states_in1 = np.empty([n_sample, 2**n_qubit], dtype=np.complex_)
    states_in2 = np.empty([n_sample, 2**n_qubit], dtype=np.complex_)
    meas_1 = np.empty([n_sample, measurement_n1], dtype=np.float_)
    meas_2 = np.empty([n_sample, measurement_n2], dtype=np.float_)
    output = np.empty([n_sample, 1])
    
    if incomplete_tomography[0]:
        fixed_states_in1 = get_subspace(n_qubit, incomplete_tomography[0], measurement_n1)
    else:
        fixed_states_in1 = [random_state(n_qubit) for _ in range(measurement_n1)]
    if incomplete_tomography[1]:
        fixed_states_in2 = get_subspace(n_qubit, incomplete_tomography[1], measurement_n2)
    else:
        fixed_states_in2 = [random_state(n_qubit) for _ in range(measurement_n2)]
    for i in range(n_sample):
        states_in1[i] = random_state(n_qubit)
        states_in2[i] = random_state(n_qubit)
        meas_1[i] = np.array([projection(s1, states_in1[i]) for s1 in fixed_states_in1])
        meas_2[i] = np.array([projection(s2, states_in2[i]) for s2 in fixed_states_in2])
        output[i, 0] = projection(states_in1[i], states_in2[i])
    result = ([meas_1, meas_2, output], [states_in1, states_in2], [fixed_states_in1, fixed_states_in2])
    if file_name is not None:
        f = gzip.open(io_paths.data_path + file_name + ".plk.gz", 'wb')
        pickle.dump(result, f, protocol=2)
        f.close()
    return result