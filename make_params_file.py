#!/usr/bin/python

import pickle


def runs():

    # Total number of trajectories
    traj_tot = 40000

    # Define the maximum time and timestep
    traj_tmax = 10.0
    traj_dt = 0.01

    # Define the Hamiltonian parameters
    bias = 1.0
    delta = 1.0
    omega_c = 2.5
    beta = 5.0
    coupling = 0.1

    # Choices for the bath types:
    # debye, ohmic, holstein, none
    bath_type = 'ohmic'

    # Define the coefficient before VB: options 1 or -1
    vb_coeff = -1.0

    # Number of oscillators for bath discretization
    nosc = 300


    # Make dictionary to contain the parameters:
    data = {
        'traj_tot' : traj_tot,
        'traj_tmax' : traj_tmax,
        'traj_dt' : traj_dt,
        'bias' : bias,
        'delta' : delta,
        'omega_c' : omega_c,
        'beta' : beta,
        'coupling' : coupling,
        'bath' : bath_type,
        'nosc' : nosc,
        'aa' : vb_coeff
    }

    make_parameter_file(data)


def make_parameter_file(data):
    """Save parameter file."""

    with open('parameters.pkl', 'wb') as f_out:
        pickle.dump(data, f_out)


if __name__ == '__main__':
   runs()
