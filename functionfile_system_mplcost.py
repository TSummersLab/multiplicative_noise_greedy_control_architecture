import numpy as np
from copy import deepcopy as dc
import pickle

import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8
# matplotlib.rcParams['image.cmap'] = 'Blues'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True


def initial_values_init(sys_in, T=100, P_max=(10**10)):

    return_values = {'T': T, 'P_max': P_max}
    # T: max time steps of Riccati iterations
    # P_max: max magnitude of cost matrix eigenvalue before assumed no convergence
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

    # W_sum = np.zeros_like(sys['W'])
    # W = dc(sys['W'])
    # if np.allclose(W, W_sum):
    #     W_check = False

    while P_check == 1:
        t += 1

        # if not W_check:
        #     W_sum += np.trace(P @ W)

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

        if np.max(np.linalg.eigvals(P_new)) > initial_values['P_max']:
            P_check = 3
            break

    J = None
    if sys['metric'] == 0:
        J = np.trace(P)
    elif sys['metric'] == 1:
        J = sys['X0'].T @ P @ sys['X0']
    elif sys['metric'] == 2:
        J = np.trace(P @ sys['X0'])

    return_values = {'P_mat': P, 'J': J, 't': t, 'P_check': P_check}
    # P_mat: cost matrix at convergence or failure
    # J: value of cost function at convergence or failure depending on the metric/initial state conditions
    # t: time of convergence or failure
    # P_check: 0 if converged, 2 if time exceeded, 3 if cost exceeded
    return return_values


################################################################

def actuator_selection_cost_1(sys_in, nu_max_in=None, initial_values=None):
    sys = dc(sys_in)

    B = sys['B']
    B_list = []
    nu_1 = None
    for i in range(0,np.shape(B)[1]):
        idx = np.argmax(B[:, i])
        if B[idx, i] > 0:
            nu_1 = i
            if idx in B_list:
                print('Duplicate actuators assigned')
                return None
            B_list.append(idx)

    cost_record = []
    time_record = []






################################################################

if __name__ == "__main__":
    print('Successfully compiled function file for system cost evaluation')
