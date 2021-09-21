import numpy as np
from copy import deepcopy as dc
import pickle

import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from functionfile_system_definition import actuator_matrix

matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8
# matplotlib.rcParams['image.cmap'] = 'Blues'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True


def initial_values_init(sys_in, T=30, P_max=(10**10)):

    return_values = {'T': T, 'P_max': P_max}
    return return_values


#################################################################

def cost_function_1(sys_in, initial_values=None):
    sys = dc(sys_in)

    A = dc(sys['A'])
    B = dc(sys['B'])

    alphai = dc(sys['alphai'])
    Ai = dc(sys['Ai'])

    betaj = dc(sys['betaj'])
    Bj = dc(sys['Bj'])

    Q = dc(sys['Q'])
    R1 = dc(sys['R1'])

    if initial_values is None:
        initial_values = initial_values_init(sys)
    else:
        initial_values = dc(initial_values)

    P = dc(Q)
    P_check = 1
    t = 0

    while P_check == 1:
        t += 1

        A_sum = np.zeros_like(A)
        if np.sum(alphai) != 0:
            for i in range(0, len(alphai)):
                A_sum += alphai[i] * (Ai[i, :, :].T @ P @ Ai[i, :, :])

        B_sum = np.zeros_like(B)
        if np.sum(betaj) != 0:
            for j in range(0, len(betaj)):
                B_sum += betaj[j] * (Bj[j, :, :].T @ P @ Bj[j, :, :])

        P_new = Q + (A.T @ P @ A) - (A.T @ P @ B @ np.linalg.inv(R1 + (B.T @ P @ B) + B_sum) @ B.T @ P @ A)

        if np.allclose(P_new, P):
            P_check = 0
            break
        else:
            P = dc(P_new)

        if t >= initial_values['T']:
            P_check = 2
            break

        if np.max(np.linalg.eigvals(P_new))>initial_values['P_max']:
            P_check = 3
            break

    return_values = {'P_mat': P, 't': t, 'P_check': P_check}
    return return_values


################################################################
