#!/usr/bin/env python

import pickle
import argparse
import time
from collections import namedtuple

import numpy as np
import numba as nb

def main(fn_parameters, fn_out, seed_rng):
    """Calculates correlation functions necessary for the construction of
        the auxiliary kernels to be used in the GQME + method X approach,
        using mixed quantum-classical method X.

        The output is an array with the following indices:
            first:  stands for 00, 01, 10c, 10p, 11c, 11p
            second: initial condition: 1, x, y, z
            third:  pauli matrix evolved in time: x, y, z
            fourth: time
        """

    print seed_rng

    # Read parameters from file
    parameters = read_parameters(fn_parameters)
    traj_tmax = parameters['traj_tmax']
    timestep = parameters['traj_dt']
    tsteps = int(traj_tmax / timestep) + 1

    # Define the time axis
    time_axis = np.arange(0, traj_tmax+timestep, timestep)

    # Define the number of subsystem states
    n_hilbert = 2
    n_liouville = n_hilbert**2

    # Allocate arrays that will contain the correlation functions
    corr_fun = np.zeros([6, n_liouville, n_liouville, tsteps], dtype='complex')
    corr_fun_sq = np.zeros_like(corr_fun)

    # Calculate the correlation functions corresponding to all initial conditions
    for i in range(n_hilbert):
        for j in range(i, n_hilbert):
            k = i * n_hilbert + j
            kkp = j * n_hilbert + i

            time0 = time.time()
            q_ij, q_ji, q_ij2, q_ji2 = calculate_trajectories(parameters, seed_rng, i, j, n_hilbert)
            time1 = time.time()
            print 'The time it took to evolve ic = |'+str(i)+'><'+str(j) + \
                  '| & |' + str(j) + '><' + str(i) + '| was {0:.4f}'.format(time1 - time0)

            corr_fun[:, k, :, :] = q_ij
            corr_fun[:, kkp, :, :] = q_ji
            corr_fun_sq[:, k, :, :] = q_ij2
            corr_fun_sq[:, kkp, :, :] = q_ji2

    # Save trajectories
    data = {'traj_time': time_axis,
            'traj': corr_fun,
            'traj2': corr_fun_sq,
            'traj_tot': parameters['traj_tot']}
    save_observables(data, fn_out)

#
# Other functions
#

def save_observables(data, fn_out):
    """Save trajectories."""

    with open(fn_out, 'wb') as f_out:
        pickle.dump(data, f_out)


def read_parameters(fn_in):
    """Read and check input parameters."""

    with open(fn_in, 'r') as f_in:
        parameters = pickle.load(f_in)

    return parameters


def construct_boson_bath(coupling, omega_c, name, nosc, bath_coeff):
    """Constructs the Wigner-transformed boson degrees of freedom."""

    if name.lower() == 'debye':
        lamb = coupling
        j = np.arange(0.5, nosc + 0.5, 1.0)
        omega_j = omega_c * np.tan(0.5 * np.pi * j / nosc)
        coupling_j = np.sqrt(2 * lamb / nosc) * omega_j
    elif name.lower() == 'ohmic':
        zeta = coupling
        j = np.arange(0.5, nosc + 0.5, 1.0)
        omega_j = - omega_c * np.log(j / nosc)
        coupling_j = np.sqrt(zeta * omega_c / nosc) * omega_j
    elif name.lower() == 'holstein':
        gamma = coupling
        omega_j = omega_c
        coupling_j = np.sqrt(2) * gamma
    elif name.lower() == 'none':
        omega_j = 1.0
        coupling_j = 0.0
    else:
        raise ValueError('Unknown bath type: {:s}'.format(name))

    coupling_j *= bath_coeff

    bath = namedtuple('bath', ('coupling_j', 'omega_j'))

    return bath(coupling_j, omega_j)


def calculate_trajectories(parameters, seed_rng, bra_ic, ket_ic, n_hilbert):
    """Calculates the correlation functions for the gqme kernel construction.
        Input:  parameters defining Hamiltonian
                seed for random number generator
                subsystem initial condition = |bra_ic> <ket_ic|

        Output: Set of correlation functions qij and qji for subsystem
                initial conditions |bra_ic> <ket_ic| and its hermitian conjugate
                Square of qij and qji for the calculation of the standard
                deviation of the error."""

    traj_tmax = parameters['traj_tmax']
    bias = parameters['bias']
    delta = parameters['delta']
    coupling = parameters['coupling']
    omega_c = parameters['omega_c']
    timestep = parameters['traj_dt']
    beta = parameters['beta']
    name_bath = parameters['bath']
    nosc = parameters['nosc']
    traj_tot = parameters['traj_tot']
    bath_coeff = parameters['aa']

    tsteps = int(traj_tmax/timestep)+1
    n_liouville = n_hilbert**2

    # Allocate the arrays for correlation functions
    q_ij = np.zeros([6, n_liouville, tsteps], dtype=complex)
    q_ji = np.zeros_like(q_ij)
    q_ij2 = np.zeros_like(q_ij)
    q_ji2 = np.zeros_like(q_ij)

    # Allocate the arrays for temporary data
    qtemp = np.empty([6, n_liouville], dtype=complex)
    qt_ij = np.empty_like(q_ij)
    qt_ji = np.empty_like(q_ij)

    # Allocate array for the time-dependent rdm
    rdm = np.empty(n_liouville, dtype=complex)

    # Allocate array for the subsystem propagator
    propagator = np.empty([n_hilbert, n_hilbert], dtype=complex)

    # Initiate random number generator
    np.random.seed(seed_rng)

    # Construct the bath
    bath = construct_boson_bath(coupling, omega_c, name_bath, nosc, bath_coeff)
    coupling_j, omega_j = bath

    # Define useful quantities for bath evolution
    bath_cos = np.cos(omega_j * 0.5 * timestep)
    bath_sin_owj = np.sin(omega_j * 0.5 * timestep) / omega_j
    bath_sin_twj = np.sin(omega_j * 0.5 * timestep) * omega_j
    cwj2 = coupling_j / (omega_j * omega_j)

    # Create some dummy arrays
    q_shifted = np.empty_like(omega_j)
    q_temp = np.empty_like(omega_j)
    p_temp = np.empty_like(omega_j)

    gg = 2 * np.tanh(0.5 * beta * omega_j)
    sigma_q = np.sqrt(1.0 / (gg * omega_j))
    sigma_p = np.sqrt(omega_j / gg)

    # Realizations of the system and bath
    for k in range(traj_tot):

        # Initialize bath
        bath_q = sigma_q * np.random.randn(nosc)
        bath_p = sigma_p * np.random.randn(nosc)

        # Calculate the bath operators needed to compute the correlation functions
        vbt = np.array([np.sum(coupling_j * bath_q)])
        vb0 = vbt[0]
        xi0 = - np.sum(coupling_j * bath_p * 0.5 * gg / omega_j)

        # Initialize subsystem
        wavefunction, ffs = initialize_subsystem_wavefunction(bra_ic, ket_ic)
        ff_ij = ffs[0]
        ff_ji = ffs[1]

        # Construct rdm from the wavefunction coefficients
        construct_RDM(wavefunction, rdm)

        # Measure final condition for correlation functions
        measure_correlation_functions(qtemp, rdm, vbt[0], vb0, xi0)

        # Measure the actual correlation functions
        qt_ij[:, :, 0] = qtemp * ff_ij
        qt_ji[:, :, 0] = qtemp * ff_ji

        # Time evolution
        for j in range(1, tsteps):

            # Update system and bath using the Ehrenfest procedure
            ehrenfest_update(bias, delta, wavefunction, cwj2, bath_q, bath_p,
                             vbt, bath_cos, bath_sin_owj, bath_sin_twj, coupling_j,
                             timestep, propagator, q_shifted, q_temp, p_temp)

            # Construct rdm from the wavefunction coefficients
            construct_RDM(wavefunction, rdm)

            # Construct the time correlation functions
            measure_correlation_functions(qtemp, rdm, vbt[0], vb0, xi0)

            # Measure the actual correlation functions
            qt_ij[:, :, j] = qtemp * ff_ij
            qt_ji[:, :, j] = qtemp * ff_ji

        # Accumulate the correlation functions for averaging
        q_ij += qt_ij
        q_ji += qt_ji
        q_ij2 += np.conj(qt_ij) * qt_ij
        q_ji2 += np.conj(qt_ji) * qt_ji

    # Average the correlation functions
    q_ij /= traj_tot
    q_ji /= traj_tot
    q_ij2 /= traj_tot
    q_ji2 /= traj_tot

    return q_ij, q_ji, q_ij2, q_ji2


def initialize_subsystem_wavefunction(bra_in, ket_in):
    """Initializes the wavefunction to a given initial condition.
        Options: population |i><i| or coherence |i><j|"""

    # TODO: Generalize to an N-level system
    # TODO: Try to avoid the if statement. It slows the thing down
    coeffs = np.zeros(2, dtype=complex)
    ic_coeffs = np.zeros(2, dtype=complex)
    if bra_in == ket_in:
        # This is a population initial condition
        random_nums = 1.0j * 2 * np.pi * np.random.rand(1)
        coeffs[bra_in] = np.exp(random_nums[0])
        # Coefficient that imposes the initial condition:
        ic_coeffs[0] = 1.0
        ic_coeffs[1] = 1.0
    else:
        # This is a coherence initial condition
        random_nums = 1.0j * 2 * np.pi * np.random.rand(2)
        sqrt2 = np.sqrt(2.)
        # Coefficients to be evolved
        coeffs[bra_in] = np.exp(random_nums[0]) / sqrt2
        coeffs[ket_in] = np.exp(random_nums[1]) / sqrt2
        # Coefficients that impose the initial condition: (i < j)
        # ic_coeff[0] corresponds to ic = |i><j|
        # ic_coeff[0] corresponds to ic = |j><i|
        ic_coeffs[0] = 4 * np.conj(coeffs[ket_in]) * coeffs[bra_in]  # note: ff2 = s21_0
        ic_coeffs[1] = np.conj(ic_coeffs[0])

    return coeffs, ic_coeffs


@nb.jit(nopython=True)
def construct_RDM(c, rdm):
    """Construct rdm from the wavefunction coefficients."""

    # TODO: Generalize this to any size for the subsystem
    rdm[0] = (np.conj(c[0]) * c[0]).real
    rdm[1] = np.conj(c[0]) * c[1]
    rdm[2] = np.conj(rdm[1])
    rdm[3] = (np.conj(c[1]) * c[1]).real


@nb.jit(nopython=True)
def measure_correlation_functions(corr_fun, rdm, vbt, vb0, xi0):
    """Records the correlation functions necessary for the auxiliary kernels."""

    corr_fun[0, :] = rdm
    corr_fun[1, :] = rdm * vbt
    corr_fun[2, :] = rdm * vb0
    corr_fun[3, :] = rdm * xi0
    corr_fun[4, :] = rdm * vb0 * vbt
    corr_fun[5, :] = rdm * xi0 * vbt


@nb.jit(nopython=True)
def ehrenfest_update(bias, delta, wavefunction, cwj2, q_bath, p_bath, vbt, \
                     bath_cos, bath_sin_owj, bath_sin_twj, coupling_j, \
                     timestep, propagator, q_shifted, q_temp, p_temp):
    """Updates system and bath EOM over full timestep
        using split operator: e^{iLt} approx e^{iL_bt/2} * e^{iL_st} * e^{iL_bt/2}
        - Analytical evolution of system EOM over full timestep
        - RK2 evolution for bath EOM over two half timesteps"""

    # Calculate the subsystem back-reaction on the bath
    system_force = cwj2 * (np.conj(wavefunction[0]) * wavefunction[0] -
                           np.conj(wavefunction[1]) * wavefunction[1]).real

    # Update the bath over a half timestep
    q_shifted[:] = q_bath + system_force
    q_temp[:] = q_shifted * bath_cos + p_bath * bath_sin_owj - system_force
    p_temp[:] = p_bath * bath_cos - q_shifted * bath_sin_twj

    # Update the bath back-reaction on the subsystem
    vbt[0] = np.sum(coupling_j * q_temp)

    # Set up analytical evolution over the subsystem
    new_bias = bias + vbt[0]
    eigenenergy = np.sqrt(new_bias**2 + delta**2)
    sin_system = -1.0j * np.sin(eigenenergy * timestep) / eigenenergy
    propagator[0, 0] = np.cos(eigenenergy * timestep) + new_bias * sin_system
    propagator[1, 1] = np.conj(propagator[0, 0])
    propagator[0, 1] = delta * sin_system
    propagator[1, 0] = propagator[0, 1]

    # Update the system wavefunction coefficients over a full timestep
    wavefunction[:] = np.dot(propagator, wavefunction)

    # Update the system back-reaction on the bath
    system_force = cwj2 * (np.conj(wavefunction[0]) * wavefunction[0] -
                           np.conj(wavefunction[1]) * wavefunction[1]).real

    # Update bath over a half timestep
    q_shifted[:] = q_temp + system_force
    q_bath[:] = q_shifted * bath_cos + p_temp * bath_sin_owj - system_force
    p_bath[:] = p_temp * bath_cos - q_shifted * bath_sin_twj


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=False, default='parameters.pkl',
                        help='Input pickle file name for parameters.')
    parser.add_argument('--output', type=str, required=False, default='trajectories.pkl',
                        help='Output pickle file name.')
    parser.add_argument('--seed_RNG', type=int, required=False, default=5,
                        help='Input the random number generator seed')
    args = parser.parse_args()

    main(args.input, args.output, args.seed_RNG)
