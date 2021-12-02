import numpy as np

import pickle

import pandas as pd

from copy import deepcopy as dc

import dataframe_image as dfi

from functionfile_system_definition import actuator_matrix_to_list, actuator_list_to_matrix, system_check, create_graph, \
    system_package, matrix_splitter

import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import matplotlib.transforms
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['legend.title_fontsize'] = 10
matplotlib.rcParams['legend.framealpha'] = 0.5
matplotlib.rcParams['lines.markersize'] = 5
# matplotlib.rcParams['image.cmap'] = 'Blues'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'


def initial_values_init(sys_in=None, T=200, P_max=(10 ** 8), P_min=(10 ** (-8))):
    return_values = {'T': T, 'P_max': P_max, 'P_min': P_min}
    if sys_in is None:
        return_values['X0'] = None
        return_values['alphai'] = None
        return_values['betaj'] = None
    else:

        sys = dc(sys_in)

        if sys['X0'].ndim == 2:
            # print('Generating random sample of initial state from given distribution for simulation')
            X0 = np.random.default_rng().multivariate_normal(mean=np.zeros(np.shape(sys['A'])[0]), cov=sys['X0'])
        else:
            X0 = dc(sys['X0'])
        return_values['X0'] = X0

        alpha_sim = np.zeros((T, len(sys['alphai'])))
        if np.sum(sys['alphai']) > 0:
            alpha_sim = np.random.default_rng().multivariate_normal(mean=np.zeros(len(sys['alphai'])),
                                                                    cov=np.diag(sys['alphai']), size=T)
        return_values['alphai'] = alpha_sim

        beta_sim = np.zeros((T, len(sys['betaj'])))
        if np.sum(sys['betaj']) > 0:
            beta_sim = np.random.default_rng().multivariate_normal(mean=np.zeros(len(sys['betaj'])),
                                                                   cov=np.diag(sys['betaj']), size=T)
        return_values['betaj'] = beta_sim

    # T: max time steps of Riccati iterations
    # P_max: max magnitude of cost matrix eigenvalue before assumed no convergence
    # P_min: tolerance of convergence of cost matrices using np.all_close()
    # alphai: simulated state-dependent noise
    # betaj: simulated input-dependent noise
    # X0: initial state simulated if distribution provided
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

    J_rec = []

    while P_check == 1:
        t += 1

        if sys['metric'] == 0:
            J_rec.append(np.trace(P))
        elif sys['metric'] == 1:
            J_rec.append(sys['X0'].T @ P @ sys['X0'])
        elif sys['metric'] == 2:
            J_rec.append(np.trace(P @ sys['X0']))

        A_sum = np.zeros_like(A)
        if np.sum(alphai) != 0:
            # print('Using alphai')
            for i in range(0, len(alphai)):
                A_sum += alphai[i] * (Ai[i, :, :].T @ P @ Ai[i, :, :])
        # else:
        #     print('NOT using alphai')

        B_sum = np.zeros_like(B)
        if np.sum(betaj) != 0:
            # print('Using betaj')
            for j in range(0, len(betaj)):
                B_sum += betaj[j] * (Bj[j, :, :].T @ P @ Bj[j, :, :])
        # else:
        #     print('NOT using betaj')

        P_new = Q + (A.T @ P @ A) + A_sum - (A.T @ P @ B @ np.linalg.inv(R1 + (B.T @ P @ B) + B_sum) @ B.T @ P @ A)

        if np.allclose(P_new, P, atol=initial_values['P_min']):
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

    B_sum = np.zeros_like(B)
    if np.sum(betaj) != 0:
        for j in range(0, len(betaj)):
            B_sum += betaj[j] * (Bj[j, :, :].T @ P @ Bj[j, :, :])
    K = -np.linalg.inv(R1 + (B.T @ P @ B) + B_sum) @ B.T @ P @ A

    return_values = {'P_mat': P, 'J_trend': J_rec, 't': t, 'P_check': P_check, 'K': K}
    # P_mat: cost matrix at convergence or failure
    # J_trend: value of cost function over iterations till failure or convergence
    # t: time of convergence or failure
    # P_check: 0 if converged, 2 if time exceeded, 3 if cost exceeded
    # K: control feedback gain for u = Kx
    return return_values


################################################################

def actuator_selection_cost_1(sys_in, nu_2=None, initial_values=None):
    # Actuator selection for state-dependent multiplicative noise
    sys = dc(sys_in)

    if initial_values is None:
        initial_values = initial_values_init(sys)

    B = sys['B']

    if nu_2 is None:
        nu_2 = np.shape(sys['B'])[1]

    S_list = []
    B_list = list(range(0, np.shape(sys['A'])[0]))
    nu_1 = None
    for i in range(0, np.shape(B)[1]):
        idx = np.argmax(B[:, i])
        if B[idx, i] > 0:
            nu_1 = i
            if idx in S_list:
                print('Duplicate actuators assigned')
                return None
            S_list.append(idx)
            B_list.remove(idx)

    if nu_1 is None:
        nu_1 = 0
    else:
        nu_1 += 1

    cost_record = []
    time_record = []
    check_record = []

    if len(B_list) == 0:
        print('Empty selection list of actuators')
        return None

    for i in range(nu_1, nu_2):
        # print('i:', i)
        cost_list = []
        time_list = []
        check_list = []
        for j in B_list:
            # print('j:', j)
            B_test = dc(B)
            B_test[j, i] = 1

            sys_test = dc(sys)
            sys_test['B'] = dc(B_test)
            if np.sum(sys_test['betaj']) > 0:
                for k in range(0, len(sys_test['betaj'])):
                    sys_test['Bj'][k] = B_test
            test_ret = cost_function_1(sys_test, initial_values)
            cost_list.append(test_ret['J_trend'][-1])
            time_list.append(test_ret['t'])
            check_list.append(test_ret['P_check'])

        idx = np.argmin(cost_list)
        if check_list[idx] != 0:
            idx = np.argmax(time_list)
        check_record.append(check_list[idx])
        cost_record.append(cost_list[idx])
        time_record.append(time_list[idx])
        B[B_list[idx], i] = 1
        B_list.remove(B_list[idx])

    sys_return = dc(sys)
    sys_return['B'] = B
    if np.sum(sys_return['betaj']) > 0:
        for k in range(0, len(sys_return['betaj'])):
            sys_return['Bj'][k] = B

    return_values = {'system': sys_return, 'cost_trend': cost_record, 'time_trend': time_record,
                     'check_trend': check_record}
    # sys: system with actuator selection
    # cost_trend: change in costs with selection of actuators
    # time_trend: change in time till convergence cost
    # check_trend: selection record
    return return_values


################################################################

def plot_actuator_selection_1(B_in, cost, time, check, fname=None):
    B = dc(B_in)

    fig1 = plt.figure(constrained_layout=True)
    gs1 = GridSpec(3, 1, figure=fig1)

    xrange = list(range(0, len(check)))
    sc1 = np.nan * np.ones(len(check))
    sc2 = np.nan * np.ones(len(check))
    t1 = np.nan * np.ones(len(check))
    t2 = np.nan * np.ones(len(check))
    for i in range(0, len(check)):
        if check[i] == 0:
            sc1[i] = cost[i]
            t1[i] = time[i]
        else:
            sc2[i] = cost[i]
            t2[i] = time[i]

    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.scatter(xrange, sc1, marker='x', color='C0')
    ax1.scatter(xrange, sc2, marker='x', color='C1')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$|S|$')
    ax1.set_ylabel(r'$J^*$')

    ax2 = fig1.add_subplot(gs1[1, 0], sharex=ax1)
    ax2.scatter(xrange, t1, marker='x', color='C0')
    ax2.scatter(xrange, t2, marker='x', color='C1')
    ax2.set_xlabel(r'$|S|$')
    ax2.set_ylabel(r'$t$')

    ax3 = fig1.add_subplot(gs1[2, 0], sharex=ax1)
    for i in range(0, np.shape(B)[1]):
        for j in range(0, np.shape(B)[0]):
            if B[j, i] == 0:
                B[j, i] = np.nan
            else:
                B[j, i:np.shape(B)[1]] = (j + 1) * np.ones(np.shape(B)[1] - i)
    # B = np.pad(B, ((0,0),(1,0)), 'constant')
    print(B)
    for i in range(0, np.shape(B)[1]):
        if check[i] == 0:
            ax3.scatter(i * np.ones(np.shape(B)[0]), B[:, i], marker='o', color='C0')
        else:
            ax3.scatter(i * np.ones(np.shape(B)[0]), B[:, i], marker='o', color='C1')
    ax3.invert_yaxis()
    ax3.set_xlabel(r'$|S|$')
    ax3.set_ylabel('Actuated Node')

    if fname is not None:
        fig1.suptitle(fname)
        fname = 'images/' + fname + '_selection.pdf'
        plt.savefig(fname, format='pdf')

    plt.show()

    return None


################################################################

def estimation_actuator_selection(sys_in, B_in=None, initial_values=None):
    sys = dc(sys_in)

    if not (B_in is None):
        sys['B'] = dc(B_in)

    if initial_values is None:
        initial_values = initial_values_init(sys)

    return_values = {'label': dc(sys['label']), 'B': sys['B']}

    for i in range(0, 1 + np.shape(sys['B'])[1]):
        sys_test = dc(sys)
        sys_test['B'] = np.zeros_like(sys['B'])
        sys_test['B'][:, 0:i] = dc(sys['B'][:, 0:i])
        if system_check(sys_test)['check']:
            # print(i, '\n', sys_test['B'])
            test_ret = cost_function_1(sys_test, initial_values)
            return_values[str(i)] = {'check': test_ret['P_check'], 'cost': test_ret['J_trend'][-1],
                                     'time': test_ret['t']}

    # return dictionary: key is number of actuators
    return return_values


################################################################

def plot_actuator_selection_2(values, fname=None):
    fig1 = plt.figure(constrained_layout=True)
    gs1 = GridSpec(3, 1, figure=fig1)

    n_vals = len(values) - 2
    B = dc(values['B'])
    x_range = list(range(0, n_vals))
    cost_check0 = np.nan * np.ones(n_vals)
    cost_check1 = np.nan * np.ones(n_vals)
    time_check0 = np.nan * np.ones(n_vals)
    time_check1 = np.nan * np.ones(n_vals)

    ax1 = fig1.add_subplot(gs1[0, 0])
    ax2 = fig1.add_subplot(gs1[1, 0], sharex=ax1)
    ax3 = fig1.add_subplot(gs1[2, 0], sharex=ax1)

    for i in range(0, 1 + np.shape(B)[1]):
        if values[str(i)]['check']:
            cost_check1[i] = values[str(i)]['cost']
            time_check1[i] = values[str(i)]['time']
        else:
            cost_check0[i] = values[str(i)]['cost']
            time_check0[i] = values[str(i)]['time']

    # print('cost_check0', cost_check0)
    # print('cost_check1', cost_check1)
    # print('time_check0', time_check0)
    # print('time_check1', time_check1)

    ax1.scatter(x_range, cost_check0, marker='x', color='C0', alpha=0.7)
    ax1.scatter(x_range, cost_check1, marker='x', color='C1', alpha=0.7)
    ax1.set_xlabel(r'$|S|$')
    ax1.set_ylabel(r'$J^*$')

    ax2.scatter(x_range, time_check0, marker='x', color='C0', alpha=0.7)
    ax2.scatter(x_range, time_check1, marker='x', color='C1', alpha=0.7)
    ax2.set_xlabel(r'$|S|$')
    ax2.set_ylabel(r'$t$')

    B = np.pad(B, ((0, 0), (1, 0)), 'constant')
    for i in range(0, np.shape(B)[1]):
        for j in range(0, np.shape(B)[0]):
            if B[j, i] == 0:
                B[j, i] = np.nan
            else:
                B[j, i:np.shape(B)[1]] = (j + 1) * np.ones(np.shape(B)[1] - i)
    # print(B)
    for i in range(0, np.shape(B)[1]):
        if values[str(i)]['check']:
            ax3.scatter(i * np.ones(np.shape(B)[0]), B[:, i], marker='o', color='C1', alpha=0.7)
        else:
            ax3.scatter(i * np.ones(np.shape(B)[0]), B[:, i], marker='o', color='C0', alpha=0.7)
    ax3.invert_yaxis()
    ax3.set_xlabel(r'$|S|$')
    # ax3.set_ylabel('Actuated Node')

    if fname is None:
        fname = values['label']

    fig1.suptitle(fname)
    fname = 'images/' + fname + '_selection.pdf'
    plt.savefig(fname, format='pdf')
    plt.show()

    return None


################################################################

def plot_actuator_selection_comparison_1(values1, values2, fname=None):
    fig1 = plt.figure(constrained_layout=True)
    gs1 = GridSpec(3, 1, figure=fig1)

    n_vals = len(values1) - 2
    B1 = dc(values1['B'])
    B2 = dc(values2['B'])
    x_range = list(range(0, n_vals))
    cost_check0 = np.nan * np.ones((n_vals, 2))
    cost_check1 = np.nan * np.ones((n_vals, 2))
    time_check0 = np.nan * np.ones((n_vals, 2))
    time_check1 = np.nan * np.ones((n_vals, 2))

    ax1 = fig1.add_subplot(gs1[0, 0])
    ax2 = fig1.add_subplot(gs1[1, 0], sharex=ax1)
    ax3 = fig1.add_subplot(gs1[2, 0], sharex=ax1)

    for i in range(0, 1 + np.shape(B1)[1]):
        if values1[str(i)]['check']:
            cost_check1[i, 0] = values1[str(i)]['cost']
            time_check1[i, 0] = values1[str(i)]['time']
        else:
            cost_check0[i, 0] = values1[str(i)]['cost']
            time_check0[i, 0] = values1[str(i)]['time']

        if values2[str(i)]['check']:
            cost_check1[i, 1] = values2[str(i)]['cost']
            time_check1[i, 1] = values2[str(i)]['time']
        else:
            cost_check0[i, 1] = values2[str(i)]['cost']
            time_check0[i, 1] = values2[str(i)]['time']

    # print('cost_check0', cost_check0)
    # print('cost_check1', cost_check1)
    # print('time_check0', time_check0)
    # print('time_check1', time_check1)

    ax1.scatter(x_range, cost_check0[:, 0], marker='x', color='C0', alpha=0.7, label=values1['label'])
    ax1.scatter(x_range, cost_check1[:, 0], marker='x', color='C2', alpha=0.7, label=values1['label'])
    ax1.scatter(x_range, cost_check0[:, 1], marker='o', color='C1', alpha=0.7, label=values2['label'])
    ax1.scatter(x_range, cost_check1[:, 1], marker='o', color='C3', alpha=0.7, label=values2['label'])
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$|S|$')
    ax1.set_ylabel(r'$J^*$')
    ax1.legend()

    ax2.scatter(x_range, time_check0[:, 0], marker='x', color='C0', alpha=0.7)
    ax2.scatter(x_range, time_check1[:, 0], marker='x', color='C2', alpha=0.7)
    ax2.scatter(x_range, time_check0[:, 1], marker='o', color='C1', alpha=0.7)
    ax2.scatter(x_range, time_check1[:, 1], marker='o', color='C3', alpha=0.7)
    ax2.set_xlabel(r'$|S|$')
    ax2.set_ylabel(r'$t$')

    B1 = np.pad(B1, ((0, 0), (1, 0)), 'constant')
    B2 = np.pad(B2, ((0, 0), (1, 0)), 'constant')
    for i in range(0, np.shape(B1)[1]):
        for j in range(0, np.shape(B1)[0]):
            if B1[j, i] == 0:
                B1[j, i] = np.nan
            else:
                B1[j, i:np.shape(B1)[1]] = (j + 1) * np.ones(np.shape(B1)[1] - i)
            if B2[j, i] == 0:
                B2[j, i] = np.nan
            else:
                B2[j, i:np.shape(B2)[1]] = (j + 1) * np.ones(np.shape(B2)[1] - i)
    # print(B)
    for i in range(0, np.shape(B1)[1]):
        if values1[str(i)]['check']:
            ax3.scatter(i * np.ones(np.shape(B1)[0]), B1[:, i], marker='1', color='C2', alpha=0.7)
        else:
            ax3.scatter(i * np.ones(np.shape(B1)[0]), B1[:, i], marker='1', color='C0', alpha=0.7)
        if values2[str(i)]['check']:
            ax3.scatter(i * np.ones(np.shape(B2)[0]), B2[:, i], marker='2', color='C3', alpha=0.7)
        else:
            ax3.scatter(i * np.ones(np.shape(B2)[0]), B2[:, i], marker='2', color='C1', alpha=0.7)
    ax3.invert_yaxis()
    ax3.set_xlabel(r'$|S|$')
    # ax3.set_ylabel('Actuated Node')

    if fname is None:
        fname = values1['label'] + ' vs ' + values2['label']

    fig1.suptitle(fname)
    fname = 'images/' + fname + '_selection.pdf'
    plt.savefig(fname, format='pdf')
    plt.show()

    return None


################################################################

def simulation_core(sys_in, feedback, initial_values=None):
    # feedback: dictionary of feedback values {'K': control gain} - constant gain over horizon
    sys = dc(sys_in)

    if initial_values is None:
        initial_values = initial_values_init(sys)

    K = dc(feedback['K'])

    nx = np.shape(sys['A'])[0]
    nu = np.shape(sys['B'])[1]

    X0 = initial_values['X0']
    T_sim = initial_values['T']
    alpha_sim = initial_values['alphai']
    beta_sim = initial_values['betaj']

    A = sys['A']
    B = sys['B']

    Ai = sys['Ai']
    Bj = sys['Bj']

    Q = sys['Q']
    R1 = sys['R1']

    state_trajectory = np.nan * np.ones((T_sim + 1, nx))
    state_trajectory[0, :] = X0

    cost_trajectory = np.nan * np.ones((T_sim + 1))
    cost_trajectory[0] = X0.T @ Q @ X0

    control_effort = np.nan * np.ones((T_sim + 1, nu))

    cost_mat = Q + (K.T @ R1 @ K)
    dyn_base_mat = A + (B @ K)

    for t in range(0, T_sim):

        cost_trajectory[t + 1] = cost_trajectory[t] + (state_trajectory[t].T @ cost_mat @ state_trajectory[t])

        dyn_noise = np.zeros((nx, nx))
        if np.sum(alpha_sim[t, :]) > 0:
            for i in range(0, np.shape(alpha_sim)[1]):
                dyn_noise += (alpha_sim[t, i] * Ai[i, :, :])
        if np.sum(beta_sim[t, :]) > 0:
            for j in range(0, np.shape(beta_sim)[1]):
                dyn_noise += (beta_sim[t, j] * (Bj[j, :, :] @ K))

        state_trajectory[t + 1, :] = ((dyn_base_mat + dyn_noise) @ state_trajectory[t, :])
        control_effort[t, :] = K @ state_trajectory[t, :]

        if np.abs(cost_trajectory[t + 1]) > initial_values['P_max']:
            print('====> Breaking current simulation at t= %s as cumulative cost magnitude exceed %.0e' % (
            t, initial_values['P_max']))
            break

    return_values = {'states': state_trajectory, 'costs': cost_trajectory, 'control': control_effort}
    # states: state trajectory for given gains
    # costs: cost trajectory along state trajectory
    # control: control effort for given trajectory
    return return_values


################################################################

def simulation_wrapper(sys_model_in, sys_true_in, initial_values=None):
    sys_model = dc(sys_model_in)
    sys_true = dc(sys_true_in)

    if initial_values is None:
        initial_values = initial_values_init(sys_true)

    # Update true system initial state covariance with realization
    # Check if true system cannot be simulated - no initial state vector/distribution provided
    if np.ndim(sys_true['X0']) == 2:
        sys_true['X0'] = dc(initial_values['X0'])
        sys_true['metric'] = 1
    elif np.allclose(sys_true['X0'], np.zeros_like(sys_true['X0'])) or sys_true['metric'] == 0:
        print('No initial state vector or distribution for true system provided - cannot run simulation')
        return None

    ret1 = cost_function_1(sys_model)
    model_feedback = {'K': dc(ret1['K'])}
    # print('Gain (K):\n', model_feedback['K'])
    ret2 = simulation_core(sys_true, model_feedback, initial_values)

    return_values = {'states': ret2['states'], 'costs': ret2['costs'], 'control': ret2['control']}
    return return_values


################################################################

def simulation_actuator_selection(sys_model_in, sys_true_in, u_low=1, initial_values=None):
    sys_model = dc(sys_model_in)
    sys_true = dc(sys_true_in)

    if initial_values is None:
        initial_values = initial_values_init(sys_true_in)

    if np.ndim(sys_true['X0']) == 2:
        sys_true['X0'] = dc(initial_values['X0'])
        sys_true['metric'] = 1
    elif np.allclose(sys_true['X0'], np.zeros_like(sys_true['X0'])) or sys_true['metric'] == 0:
        print('No initial state vector or distribution for true system provided - cannot run simulation')
        return None

    if np.shape(sys_model['B']) != np.shape(sys_true['B']):
        print('Controllers are not the same structure')
        return None

    states = {}
    costs = {}
    control = {}

    nx = np.shape(sys_model['A'])[0]
    for i in range(u_low, 1 + np.shape(sys_model['B'])[1]):
        sys_model_test = dc(sys_model)
        sys_true_test = dc(sys_true)

        B_test1 = np.zeros_like(sys_model['B'])
        B_test2 = np.zeros_like(sys_true['B'])

        B_test1[:, 0:i] = dc(sys_model['B'][:, 0:i])
        sys_model_test['B'] = dc(B_test1)
        B_test2[:, 0:i] = dc(sys_model['B'][:, 0:i])
        sys_true_test['B'] = dc(B_test2)

        if np.sum(sys_model_test['betaj']) > 0:
            for j in range(0, len(sys_model_test['betaj'])):
                sys_model_test['Bj'][j, :, :] = dc(B_test1)
        if np.sum(sys_true_test['betaj']) > 0:
            for j in range(0, len(sys_true_test['betaj'])):
                sys_true_test['Bj'][j, :, :] = dc(B_test2)

        test_return = simulation_wrapper(sys_model_test, sys_true_test, initial_values)
        states[str(i)] = test_return['states']
        costs[str(i)] = test_return['costs']
        control[str(i)] = test_return['control']

    return_values = {'label': sys_model['label'] + ' ' + sys_true['label'], 'states': states, 'costs': costs,
                     'control': control}
    # states: dictionary of state trajectories for each actuator size
    # costs: dictionary of cost trends for each actuator size
    # control: dictionary of control effort for each actuator size
    return return_values


################################################################

def simulation_model_comparison(sys_modelA_in, sys_modelB_in, sys_true_in, initial_values=None):
    print('Simulation start: Comparison of actuator selection of A vs B on C')
    sys_true = dc(sys_true_in)
    sys_modelA = dc(sys_modelA_in)
    sys_modelB = dc(sys_modelB_in)

    if initial_values is None:
        initial_values = initial_values_init(sys_true)

    sys_modelA = dc(actuator_selection_cost_1(sys_modelA, initial_values=initial_values)['system'])
    sys_modelB = dc(actuator_selection_cost_1(sys_modelB, initial_values=initial_values)['system'])

    ret_A = simulation_actuator_selection(sys_modelA, sys_true, initial_values=initial_values)
    ret_B = simulation_actuator_selection(sys_modelB, sys_true, initial_values=initial_values)

    ret_A['label'] = sys_modelA['label'] + ' on ' + sys_true['label']
    ret_B['label'] = sys_modelA['label'] + ' on ' + sys_true['label']

    return_values = {'T_A': ret_A, 'T_B': ret_B, 'system_A': sys_modelA, 'system_B': sys_modelB, 'system_C': sys_true}
    # T_A, T_B: state, input and cost trajectories of simulations of true system using A and B respectively
    # system_A, system_B: systems with optimal selection of actuator sets
    print('Simulation end: Comparison of actuator selection of A vs B on C')
    return return_values


# ################################################################
#
# def simulation_2_model_comparison(sys_base_in, sys_model1_in, sys_model2_in, initial_values=None):
#     sys_base = dc(sys_base_in)
#     sys_model1 = dc(sys_model1_in)
#     sys_model2 = dc(sys_model2_in)
#
#     if initial_values is None:
#         initial_values = initial_values_init(sys_true)
#
#     sys_model1 = dc(actuator_selection_cost_1(sys_model1, initial_values=initial_values)['system'])
#     sys_model2 = dc(actuator_selection_cost_1(sys_model2, initial_values=initial_values)['system'])
#
#     ret_1 = simulation_actuator_selection(sys_model1, sys_base, initial_values=initial_values)
#     ret_2 = simulation_actuator_selection(sys_model2, sys_base, initial_values=initial_values)
#
#     ret_1['label'] = sys_model1['label'] + 'from Model 1'
#     ret_2['label'] = sys_model2['label'] + 'from Model 2'
#
#     return_values = {'T_Nom': ret_nom, 'T_MPL': ret_mpl, 'system_mpl': sys_mpl, 'system_nom': sys_nom}
#     return return_values


################################################################

def plot_simulation(display_data=None, T=None, fname=None):
    if display_data is None:
        print('No data provided')
        return None

    fig1 = plt.figure(constrained_layout=True)
    gs1 = GridSpec(3, 1, figure=fig1)

    if T is None:
        if 'states' in display_data:
            s = 'states'
        elif 'costs' in display_data:
            s = 'costs'
        elif 'control' in display_data:
            s = 'control'
        else:
            print('Error')
            return None
        for i in display_data[s]:
            T = np.shape(display_data[s][i])[0]
            break
    T_range = list(range(0, T))

    if 'states' in display_data:
        ax1 = fig1.add_subplot(gs1[0, 0])
        for i in display_data['states']:
            ax1.plot(T_range, display_data['states'][i], color='C' + i, alpha=0.5, label=i)
        # ax1.set_yscale('log')
        ax1.set_xlabel(r'$t$')
        ax1.set_ylabel(r'$x_t$')

    if 'costs' in display_data:
        ax2 = fig1.add_subplot(gs1[1, 0])
        for i in display_data['costs']:
            ax2.plot(T_range, display_data['costs'][i], color='C' + i, alpha=0.5, label=i)
        ax2.set_xlabel(r'$t$')
        ax2.set_ylabel(r'$J^*$')
        ax2.set_yscale('log')
        ax2.legend(ncol=3)

    if 'control' in display_data:
        ax3 = fig1.add_subplot(gs1[2, 0])
        for i in display_data['control']:
            ax3.plot(T_range, display_data['control'][i], color='C' + i, alpha=0.5, label=i)
        ax3.set_xlabel(r'$t$')
        ax3.set_ylabel(r'$u_t$')

    if fname is None:
        fname = display_data['label']

    fig1.suptitle(fname)
    fname = 'images/' + fname + '.pdf'
    plt.savefig(fname, format='pdf')
    plt.show()

    return None


################################################################

def plot_simulation_comparison1(values):
    valuesA = dc(values['T_A'])
    valuesB = dc(values['T_B'])

    fig1 = plt.figure(constrained_layout=True)
    gs1 = GridSpec(3, 1, figure=fig1)

    T = np.shape(valuesA['states']['1'])[0]
    T_range = list(range(0, T))

    ax1 = fig1.add_subplot(gs1[0, 0])
    for key in valuesA['states']:
        # ax1.plot(T_range, valuesA['states'][key], marker='o', alpha=0.5, color='C'+key)
        # ax1.plot(T_range, valuesB['states'][key], marker='x', alpha=0.5, color='C'+key)
        ax1.plot(T_range, valuesA['states'][key], linewidth=1, alpha=0.5, color='C' + key)
        ax1.plot(T_range, valuesB['states'][key], ls='-.', linewidth=2, alpha=0.5, color='C' + key)
    # ax1.set_xlabel(r'$t$')
    ax1.xaxis.set_tick_params(labelbottom=False)
    ax1.set_ylabel(r'$x_t$')

    ax2 = fig1.add_subplot(gs1[1, 0], sharex=ax1)
    for key in valuesA['control']:
        # ax2.plot(T_range, valuesA['control'][key], marker='o', alpha=0.5, color='C'+key)
        # ax2.plot(T_range, valuesB['control'][key], marker='x', alpha=0.5, color='C'+key)
        ax2.plot(T_range, valuesA['control'][key], linewidth=1, alpha=0.5, color='C' + key)
        ax2.plot(T_range, valuesB['control'][key], ls='-.', linewidth=2, alpha=0.5, color='C' + key)
    # ax2.set_xlabel(r'$t$')
    ax2.xaxis.set_tick_params(labelbottom=False)
    ax2.set_ylabel(r'$u_t$')

    ax3 = fig1.add_subplot(gs1[2, 0], sharex=ax1)
    for key in valuesA['costs']:
        # ax3.plot(T_range, valuesA['costs'][key], marker='o', markeredgewidth=0.5, alpha=0.5, color='C'+key)
        # ax3.plot(T_range, valuesB['costs'][key], marker='x', markeredgewidth=0.5, alpha=0.5, color='C'+key)
        ax3.plot(T_range, valuesA['costs'][key], linewidth=1, alpha=0.5, color='C' + key, label='A:' + key)
        ax3.plot(T_range, valuesB['costs'][key], ls='-.', linewidth=2, alpha=0.5, color='C' + key, label='B:' + key)
        # ax3.plot(T_range, valuesA['costs'][key], ls=':', marker='x', alpha=0.5, color='C' + key)
        # ax3.plot(T_range, valuesB['costs'][key], alpha=0.5, color='C' + key)
    ax3.set_xlabel(r'$t$')
    ax3.set_ylabel(r'$J_t$')
    ax3.set_yscale('log')
    ax3.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.5), title=r'Model:$|S|$')

    try:
        fname = 'images/' + values['file_name'] + '_comparison1.pdf'
        plt.savefig(fname, format='pdf')
        print('Plot saved as %s' % fname)
    except:
        print('Plot not saved')

    plt.show()

    return None


################################################################

def plot_simulation_comparison2(values):
    valuesA = dc(values['T_A'])
    valuesB = dc(values['T_B'])

    fig1 = plt.figure(constrained_layout=True)
    gs1 = GridSpec(2, 1, figure=fig1)

    T = np.shape(valuesA['costs']['1'])[0]
    T_range = list(range(0, T))

    ax1 = fig1.add_subplot(gs1[0, 0])
    for key in valuesA['costs']:
        ax1.plot(T_range, valuesA['costs'][key], linewidth=1, alpha=0.5, color='C' + key)
        ax1.plot(T_range, valuesB['costs'][key], ls='-.', linewidth=2, alpha=0.5, color='C' + key)
        # ax1.plot(T_range, valuesA['costs'][key], marker='o', alpha=0.5, color='C' + key)
        # ax1.plot(T_range, valuesB['costs'][key], marker='x', alpha=0.5, color='C' + key)
        # ax1.plot(T_range, valuesA['costs'][key], ls=':', marker='x', alpha=0.5, color='C' + key)
        # ax1.plot(T_range, valuesB['costs'][key], alpha=0.5, color='C' + key)
    # ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$J_t$')
    ax1.set_yscale('log')

    ax2 = fig1.add_subplot(gs1[1, 0], sharex=ax1)
    for key in valuesA['costs']:
        ax2.plot(T_range, valuesA['costs'][key] - valuesB['costs'][key], alpha=0.5, color='C' + key, label=key)
    ax2.set_xlabel(r'$t$')
    ax2.set_ylabel(r'$J_t$ (A - B)')
    ax2.set_yscale('log')
    ax2.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.5), title=r'$|S|$')

    try:
        fname = 'images/' + values['file_name'] + '_comparison2.pdf'
        plt.savefig(fname, format='pdf')
        print('Plot saved as %s' % fname)
    except:
        print('Plot not saved')

    plt.show()

    return None


################################################################

# def plot_simulation_nom_vs_mpl_1(values, fname=None):
# 
#     Nom_values = dc(values['T_Nom'])
#     MPL_values = dc(values['T_MPL'])
# 
#     fig1 = plt.figure(constrained_layout=True)
#     gs1 = GridSpec(3, 1, figure=fig1)
# 
#     T = np.shape(Nom_values['states']['1'])[0]
#     T_range = list(range(0, T))
# 
#     ax1 = fig1.add_subplot(gs1[0, 0])
#     for key in Nom_values['states']:
#         # ax1.plot(T_range, Nom_values['states'][key], marker='o', alpha=0.5, color='C'+key)
#         # ax1.plot(T_range, MPL_values['states'][key], marker='x', alpha=0.5, color='C'+key)
#         ax1.plot(T_range, Nom_values['states'][key], linewidth=1, alpha=0.5, color='C' + key)
#         ax1.plot(T_range, MPL_values['states'][key], ls='-.', linewidth=2, alpha=0.5, color='C' + key)
#     ax1.set_xlabel(r'$t$')
#     ax1.set_ylabel(r'$x_t$')
# 
#     ax2 = fig1.add_subplot(gs1[1, 0], sharex=ax1)
#     for key in Nom_values['control']:
#         # ax2.plot(T_range, Nom_values['control'][key], marker='o', alpha=0.5, color='C'+key)
#         # ax2.plot(T_range, MPL_values['control'][key], marker='x', alpha=0.5, color='C'+key)
#         ax2.plot(T_range, Nom_values['control'][key], linewidth=1, alpha=0.5, color='C' + key)
#         ax2.plot(T_range, MPL_values['control'][key], ls='-.', linewidth=2, alpha=0.5, color='C' + key)
#     ax2.set_xlabel(r'$t$')
#     ax2.set_ylabel(r'$u_t$')
# 
#     ax3 = fig1.add_subplot(gs1[2, 0], sharex=ax1)
#     for key in Nom_values['costs']:
#         # ax3.plot(T_range, Nom_values['costs'][key], marker='o', markeredgewidth=0.5, alpha=0.5, color='C'+key)
#         # ax3.plot(T_range, MPL_values['costs'][key], marker='x', markeredgewidth=0.5, alpha=0.5, color='C'+key)
#         ax3.plot(T_range, Nom_values['costs'][key], linewidth=1, alpha=0.5, color='C' + key, label='Nom:'+key)
#         ax3.plot(T_range, MPL_values['costs'][key], ls='-.', linewidth=2, alpha=0.5, color='C' + key, label='MPL:'+key)
#         # ax3.plot(T_range, Nom_values['costs'][key], ls=':', marker='x', alpha=0.5, color='C' + key)
#         # ax3.plot(T_range, MPL_values['costs'][key], alpha=0.5, color='C' + key)
#     ax3.set_xlabel(r'$t$')
#     ax3.set_ylabel(r'$J_t$')
#     ax3.set_yscale('log')
#     ax3.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.5), title=r'Model:$|S|$')
#     plt.show()
# 
#     return None
# 
# def plot_simulation_nom_vs_mpl_2(values, fname=None):
#     Nom_values = dc(values['T_Nom'])
#     MPL_values = dc(values['T_MPL'])
# 
#     fig1 = plt.figure(constrained_layout=True)
#     gs1 = GridSpec(2, 1, figure=fig1)
# 
#     T = np.shape(Nom_values['costs']['1'])[0]
#     T_range = list(range(0, T))
# 
#     ax1 = fig1.add_subplot(gs1[0, 0])
#     for key in Nom_values['costs']:
#         ax1.plot(T_range, Nom_values['costs'][key], linewidth=1, alpha=0.5, color='C' + key)
#         ax1.plot(T_range, MPL_values['costs'][key], ls='-.', linewidth=2, alpha=0.5, color='C' + key)
#         # ax1.plot(T_range, Nom_values['costs'][key], marker='o', alpha=0.5, color='C' + key)
#         # ax1.plot(T_range, MPL_values['costs'][key], marker='x', alpha=0.5, color='C' + key)
#         # ax1.plot(T_range, Nom_values['costs'][key], ls=':', marker='x', alpha=0.5, color='C' + key)
#         # ax1.plot(T_range, MPL_values['costs'][key], alpha=0.5, color='C' + key)
#     ax1.set_xlabel(r'$t$')
#     ax1.set_ylabel(r'$J_t$')
#     ax1.set_yscale('log')
# 
#     ax2 = fig1.add_subplot(gs1[1, 0], sharex=ax1)
#     for key in Nom_values['costs']:
#         ax2.plot(T_range, Nom_values['costs'][key]-MPL_values['costs'][key], alpha=0.5, color='C'+key, label=key)
#     ax2.set_xlabel(r'$t$')
#     ax2.set_ylabel(r'$J_t$ (Nominal - MPL feedback)')
#     ax2.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.5), title=r'$|S|$')
# 
#     plt.show()
# 
#     return None
# 
# 
# ################################################################


def actuator_comparison(values, disptext=True, figplt=True):
    SA = dc(values['system_A'])
    SB = dc(values['system_B'])
    # def actuator_comparison(sysA_in, sysB_in, disptext=True, figplt=True):
    #     SA = dc(sysA_in)
    #     SB = dc(sysB_in)

    return_value = {}

    if np.allclose(SA['B'], SB['B']):
        if disptext:
            print('Both control sets are close/equal')
        return_value['act_comp'] = 0
        if figplt:
            fig1 = plt.figure(constrained_layout=True)
            gs1 = GridSpec(1, 1, figure=fig1)
            ax1 = fig1.add_subplot(gs1[0, 0])
            B = dc(SA['B'])
            for i in range(0, np.shape(B)[1]):
                for j in range(0, np.shape(B)[0]):
                    if B[j, i] == 0:
                        B[j, i] = np.nan
                    else:
                        B[j, i:np.shape(B)[1]] = (j + 1) * np.ones(np.shape(B)[1] - i)
                ax1.scatter((i + 1) * np.ones(np.shape(B)[0]), B[:, i], marker='o', color='C0', s=100)
            # ax1.set_ylim(bottom=1-0.5, top=np.shape(B)[0]+0.5)
            # ax1.set_xlim(left=1-0.5, right=np.shape(B)[1]+0.5)
            ax1.invert_yaxis()
            ax1.set_xlabel(r'$|S|$')
            ax1.set_ylabel('Node number')
            ax1.set_title('Actuator Set comparison')
        else:
            if disptext:
                print(SA['label'], ' B = ', SB['label'], ' B:\n', SB['B'])

    else:
        if disptext:
            print('Control sets are different')
        return_value['act_comp'] = 1
        if figplt:
            fig1 = plt.figure(constrained_layout=True)
            gs1 = GridSpec(1, 1, figure=fig1)
            ax1 = fig1.add_subplot(gs1[0, 0])
            B1 = SA['B']
            B2 = SB['B']
            for i in range(0, np.shape(B1)[1]):
                for j in range(0, np.shape(B1)[0]):
                    if B1[j, i] == 0:
                        B1[j, i] = np.nan
                    else:
                        B1[j, i:np.shape(B1)[1]] = (j + 1) * np.ones(np.shape(B1)[1] - i)
                    if B2[j, i] == 0:
                        B2[j, i] = np.nan
                    else:
                        B2[j, i:np.shape(B2)[1]] = (j + 1) * np.ones(np.shape(B2)[1] - i)
                ax1.scatter((i + 1) * np.ones(np.shape(B1)[0]), B1[:, i], marker='1', color='C0', s=100)
                ax1.scatter((i + 1) * np.ones(np.shape(B2)[0]), B2[:, i], marker='2', color='C1', s=100)
            # ax1.set_ylim(bottom=1-0.5, top=np.shape(B1)[0]+0.5)
            # ax1.set_xlim(left=1-0.5, right=np.shape(B1)[1]+0.5)
            ax1.invert_yaxis()
            ax1.set_xlabel(r'$|S|$')
            ax1.set_ylabel('Node number')
            ax1.set_title('Actuator Set comparison')
            ax1.legend(['A', 'B'])
        else:
            if disptext:
                print('System A', ' B:\n', SA['B'])
                print('System B', ' B:\n', SB['B'])
                print('B diff (%s - %s):' % (SA['label'], SB['label']))
                print(SA['B'] - SB['B'])

    # print(SA['label'], ' B:\n', SA['B'])
    # print(SB['label'], ' B:\n', SB['B'])
    if figplt:
        try:
            fname = 'images/' + values['file_name'] + '_actcomparison.pdf'
            plt.savefig(fname, format='pdf')
            print('Plot saved as %s' % fname)
            plt.show()
        except:
            print('Plot not saved')

    return return_value


################################################################

def random_graph_empirical_simulation(sys_model, network_parameter, network_type, number_of_iterations=50,
                                      save_check=True):
    if network_type != 'ER' and network_type != 'BA':
        print('ERROR: Check network model')
        return None

    print('\nSimulation start: Empirical study of random graphs\n')
    print('Model: ' + network_type + '\n')

    nx = np.shape(sys_model['A'])[0]
    rho = round(np.max(np.abs(np.linalg.eigvals(sys_model['A']))), 1)
    alphai = dc(sys_model['alphai'])
    X0 = dc(sys_model['X0'])

    N_test = number_of_iterations

    cost_record_A = np.nan * np.zeros((N_test, nx))
    cost_record_B = np.nan * np.zeros((N_test, nx))

    for iter in range(0, N_test):

        print("Realization: %s / %s" % (iter + 1, N_test))

        if network_type == 'ER':
            RG1 = create_graph(nx, type='ER', p=network_parameter)
            RG2 = create_graph(nx, type='ER', p=network_parameter)
            RG3 = create_graph(nx, type='ER', p=network_parameter)
        elif network_type == 'BA':
            RG1 = create_graph(nx, type='BA', p=network_parameter)
            RG2 = create_graph(nx, type='BA', p=network_parameter)
            RG3 = create_graph(nx, type='BA', p=network_parameter)
        else:
            print('ERROR: Specify network model')
            return None

        S_A = system_package(A_in=rho * RG1['A'], X0_in=X0, label_in='System A', print_check=False)
        if not system_check(S_A)['check']:
            print('System A Error')

        Ai_B = matrix_splitter(np.abs(RG3['Adj'] - RG1['Adj']))
        alphai_B = alphai * np.ones(np.shape(Ai_B)[0])
        S_B = system_package(A_in=rho * RG1['A'], alphai_in=alphai_B, Ai_in=Ai_B, X0_in=X0, label_in='System B',
                             print_check=False)
        if not system_check(S_B)['check']:
            print('System B Error')

        S_C = system_package(A_in=rho * RG3['A'], X0_in=X0, alphai_in=alphai, Ai_in=RG2['Adj'], label_in='System C',
                             print_check=False)
        if not system_check(S_C)['check']:
            print('True System Error')

        ret_sim = simulation_model_comparison(S_A, S_B, S_C)

        for i in ret_sim['T_A']['costs']:
            cost_record_A[iter, int(i) - 1] = ret_sim['T_A']['costs'][i][-1]
        for i in ret_sim['T_B']['costs']:
            cost_record_B[iter, int(i) - 1] = ret_sim['T_B']['costs'][i][-1]

    return_values = {'A_costs': cost_record_A, 'B_costs': cost_record_B, 'network_parameter': network_parameter,
                     'nx': nx, 'network_type': network_type, 'N_test': N_test, 'rho': rho, 'alphai': alphai}
    print('\nSimulation end: Empirical study of random graphs\n')

    if save_check:
        try:
            fname = 'system_test/MPL_' + str(return_values['N_test']) + '_' + return_values['network_type'] + '_' + str(
                return_values['network_parameter']) + '_' + str(return_values['nx']) + '_' + str(
                return_values['rho']) + '_' + str(return_values['alphai'][0]) + '.pickle'
            f_open = open(fname, 'wb')
            pickle.dump(return_values, f_open)
            f_open.close()
            print('System saved to file @', fname, '\n')
        except:
            print('Save failed')

    return return_values


################################################################

def random_graph_empirical_simulation_read(sys_model, network_parameter, network_type, number_of_iterations=50):
    N_test = number_of_iterations
    nx = np.shape(sys_model['A'])[0]
    rho = round(np.max(np.linalg.eigvals(sys_model['A'])), 1)
    alphai = sys_model['alphai']
    fname = 'system_test/MPL_' + str(N_test) + '_' + network_type + '_' + str(network_parameter) + '_' + str(
        nx) + '_' + str(rho) + '_' + str(alphai[0]) + '.pickle'

    try:
        f_open = open(fname, 'rb')
    except:
        print('ERROR: File @', fname, ' does not exist or not readable')
        return None

    return_values = pickle.load(f_open)
    f_open.close()
    print('System read from file @', fname, '\n')
    return return_values


################################################################

def random_graph_empirical_simulation_comparison(N_test, Network_type, Network_parameter, nx, alphai, rho):
    simulation_values = {}
    parameter_list = {}

    if type(Network_parameter) is list:
        parameter_list['p_list'] = Network_parameter
        parameter_list['p_type'] = 'p'
        for i in range(0, len(Network_parameter)):
            S_base_model = system_package(A_in=rho * create_graph(nx)['A'], alphai_in=alphai,
                                          Ai_in=create_graph(nx)['A'], label_in='System Model', print_check=False)
            simulation_values[i] = random_graph_empirical_simulation_read(S_base_model, Network_parameter[i],
                                                                          Network_type, N_test)
    elif type(nx) is list:
        parameter_list['p_list'] = nx
        parameter_list['p_type'] = 'nx'
        for i in range(0, len(nx)):
            S_base_model = system_package(A_in=rho * create_graph(nx[i])['A'], alphai_in=alphai,
                                          Ai_in=create_graph(nx[i])['A'], label_in='System Model', print_check=False)
            simulation_values[i] = random_graph_empirical_simulation_read(S_base_model, Network_parameter, Network_type,
                                                                          N_test)
    else:
        print('ERROR: Check comparison list')
        return None

    return simulation_values, parameter_list


################################################################

def plot_random_graph_simulation(plt_data):
    Nom_values = []
    MPL_values = []

    Nom_check_fail = []
    MPL_check_fail = []

    Nom_check_pass = []
    MPL_check_pass = []

    Nom_mean = []
    MPL_mean = []

    Nom_pos = []  # list(range(1, 1 + np.shape(plt_data['A_costs'])[1]))
    MPL_pos = []  # list(range(1, 1 + np.shape(plt_data['B_costs'])[1]))

    x_range = list(range(1, 1 + plt_data['nx'], 2))

    for i in range(0, plt_data['nx']):
        Nom_values.append([j for j in plt_data['A_costs'][:, i] if not np.isnan(j)])
        Nom_check_fail.append(np.sum(np.isnan(plt_data['A_costs'][:, i])))
        if len(Nom_values[-1]) > 0:
            Nom_pos.append(i + 1)
        else:
            del Nom_values[-1]

    for i in range(0, plt_data['nx']):
        MPL_values.append([j for j in plt_data['B_costs'][:, i] if not np.isnan(j)])
        MPL_check_fail.append(np.sum(np.isnan(plt_data['B_costs'][:, i])))
        if len(MPL_values[-1]) > 0:
            MPL_pos.append(i + 1)
        else:
            del MPL_values[-1]

    for i in range(0, len(Nom_check_fail)):
        Nom_check_pass.append(plt_data['N_test'] - Nom_check_fail[i])
        # if Nom_check_fail[i] == 0:
        #     Nom_check_fail[i] = np.nan
    for i in range(0, len(MPL_check_fail)):
        MPL_check_pass.append(plt_data['N_test'] - MPL_check_fail[i])
        # if MPL_check_fail[i] == 0:
        #     MPL_check_fail[i] = np.nan

    fig1 = plt.figure(constrained_layout=True)
    gs1 = GridSpec(3, 1, figure=fig1)
    ax1 = fig1.add_subplot(gs1[0:2, 0])
    # for i in range(0, np.shape(plt_data['A_costs'])[0]):
    #     ax1.violinplot(plt_data['A_costs'][i, ~np.isnan(plt_data['A_costs'][i])], i+1, showmeans=True)
    # for i in range(0, np.shape(plt_data['B_costs'])[0]):
    #     ax1.violinplot(plt_data['B_costs'][i, ~np.isnan(plt_data['B_costs'][i])], i+1, showmeans=True)
    ax1.violinplot(Nom_values, Nom_pos, showmeans=True)
    ax1.violinplot(MPL_values, MPL_pos, showmeans=True)
    ax1.set_xticks(x_range)

    mean_nom = [np.mean(i) for i in Nom_values]
    mean_mpl = [np.mean(i) for i in MPL_values]

    ax1.plot(Nom_pos, mean_nom, color='C0', label='A')
    ax1.plot(MPL_pos, mean_mpl, color='C3', label='B')
    ax1.set_yscale('log')
    ax1.set_ylabel('Cost')
    ax1.legend()

    adjust = 0.1
    xA_pos = np.linspace(1 - adjust, plt_data['nx'] - adjust, plt_data['nx'])
    xB_pos = np.linspace(1 + adjust, plt_data['nx'] + adjust, plt_data['nx'])
    y_tick_vals = range(0, plt_data['N_test'] + 1, 20)

    ax2 = fig1.add_subplot(gs1[2, 0], sharex=ax1)
    ax2.bar(xA_pos, Nom_check_pass, width=2 * adjust, color='C0', label='A Pass', edgecolor='k', linewidth=0.5,
            alpha=0.7)
    ax2.bar(xA_pos, Nom_check_fail, width=2 * adjust, bottom=Nom_check_pass, color='C1', label='A Fail', edgecolor='k',
            linewidth=0.5, alpha=0.7)
    ax2.bar(xB_pos, MPL_check_pass, width=2 * adjust, color='C2', label='B Pass', edgecolor='k', linewidth=0.5,
            alpha=0.7)
    ax2.bar(xB_pos, MPL_check_fail, width=2 * adjust, bottom=MPL_check_pass, color='C3', label='B Fail', edgecolor='k',
            linewidth=0.5, alpha=0.7)

    # ax2.scatter(range(1, 1+len(Nom_check_fail)), Nom_check_fail, alpha=0.5, color='C0', label='A')
    # ax2.scatter(range(1, 1+len(MPL_check_fail)), MPL_check_fail, alpha=0.5, color='C3', label='B')
    ax2.legend(ncol=2, loc='lower right')
    ax2.set_xlabel(r'$|S|$')
    ax2.set_ylabel('Control Check for\n' + str(np.shape(plt_data['B_costs'])[0]) + ' models')
    ax2.set_xticks(x_range)
    ax2.set_yticks(y_tick_vals)

    fname = 'images/MPL_' + str(plt_data['N_test']) + '_' + plt_data['network_type'] + '_' + str(
        plt_data['network_parameter']) + '_' + str(plt_data['nx']) + '_' + str(plt_data['rho']) + '_Comp1.pdf'
    try:
        plt.savefig(fname, format='pdf')
        print('File saved as: %s' % (fname))
    except:
        print('Save failed')
    plt.show()

    return None


################################################################

def plot_random_graph_simulation2(plt_data):
    A_costs = dc(plt_data['A_costs'])
    B_costs = dc(plt_data['B_costs'])

    A_costs = np.where(np.isnan(A_costs), np.inf, A_costs)
    B_costs = np.where(np.isnan(B_costs), np.inf, B_costs)

    A_values = []
    B_values = []

    A_check_fail = []
    B_check_fail = []

    A_check_pass = []
    B_check_pass = []

    A_mean = []
    B_mean = []

    A_median = []
    B_median = []

    A_pos = []
    B_pos = []

    # Use np.where to swap nan with inf - check for mean and median calculations that way

    x_range = list(range(1, 1 + plt_data['nx'], 2))

    for i in range(0, plt_data['nx']):

        A_values.append([j for j in A_costs[:, i] if not np.isinf(j)])
        A_check_fail.append(np.sum(np.isinf(A_costs[:, i])))
        A_check_pass.append(plt_data['N_test'] - A_check_fail[-1])
        A_mean.append(np.mean(A_costs[:, i]))
        A_median.append(np.median(A_costs[:, i]))
        if len(A_values[-1]) == 0:
            del A_values[-1]
        else:
            A_pos.append(i + 1)

        B_values.append([j for j in B_costs[:, i] if not np.isinf(j)])
        B_check_fail.append(np.sum(np.isinf(B_costs[:, i])))
        B_check_pass.append(plt_data['N_test'] - B_check_fail[-1])
        B_mean.append(np.mean(B_costs[:, i]))
        B_median.append(np.median(B_costs[:, i]))
        if len(B_values[-1]) == 0:
            del B_values[-1]
        else:
            B_pos.append(i + 1)

    A_check_pass = [a / plt_data['N_test'] for a in A_check_pass]
    # A_check_fail = [a / plt_data['N_test'] for a in A_check_fail]

    B_check_pass = [a / plt_data['N_test'] for a in B_check_pass]
    # B_check_fail = [a / plt_data['N_test'] for a in B_check_fail]

    x_val = range(1, 1 + plt_data['nx'])

    fig1 = plt.figure(constrained_layout=True)
    gs1 = GridSpec(2, 1, figure=fig1)

    ax1 = fig1.add_subplot(gs1[0, 0])

    ax1.plot(x_val, A_mean, color='C0', marker='o', alpha=0.7, label='mean(A)')
    ax1.plot(x_val, B_mean, color='C3', marker='o', alpha=0.7, label='mean(B)')

    ax1.plot(x_val, A_median, color='C0', marker='o', alpha=0.7, label='median(A)', linestyle='dashed')
    ax1.plot(x_val, B_median, color='C3', marker='o', alpha=0.7, label='median(B)', linestyle='dashed')

    ax1.set_xticks(x_range)
    ax1.xaxis.set_tick_params(labelbottom=False)
    ax1.set_yscale('log')
    ax1.set_ylabel('Cost')
    ax1.grid(visible=True, which='both', axis='x', color='k', linestyle='dotted', linewidth=0.5, alpha=0.5)
    ax1.legend(ncol=2, loc='best')
    # ax2.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.5), title=r'$|S|$')

    # adjust = 0.1
    # xA_pos = np.linspace(1 - adjust, plt_data['nx'] - adjust, plt_data['nx'])
    # xB_pos = np.linspace(1 + adjust, plt_data['nx'] + adjust, plt_data['nx'])
    y_tick_vals = [0, 0.5, 1]

    ax2 = fig1.add_subplot(gs1[1, 0], sharex=ax1)
    # ax2.bar(xA_pos, A_check_pass, width=2 * adjust, color='C0', label='A Pass', edgecolor='k', linewidth=0.5, alpha=0.7)
    # ax2.bar(xA_pos, A_check_fail, width=2 * adjust, bottom=A_check_pass, color='C1', label='A Fail', edgecolor='k', linewidth=0.5, alpha=0.7)
    # ax2.bar(xB_pos, B_check_pass, width=2 * adjust, color='C2', label='B Pass', edgecolor='k', linewidth=0.5, alpha=0.7)
    # ax2.bar(xB_pos, B_check_fail, width=2 * adjust, bottom=B_check_pass, color='C3', label='B Fail', edgecolor='k', linewidth=0.5, alpha=0.7)

    ax2.plot(x_val, A_check_pass, color='C0', marker='o', label='A', alpha=0.7)
    ax2.plot(x_val, B_check_pass, color='C3', marker='o', label='B', alpha=0.7)

    ax2.legend(ncol=2, loc='lower right')
    ax2.set_xlabel(r'$|S|$')
    ax2.set_ylabel('Control Pass Fraction \n(' + str(plt_data['N_test']) + ' Realizations)')
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4, integer=True))
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(color='k', linestyle='dotted', linewidth=0.5, alpha=0.5)
    ax2.grid(visible=True, which='both', axis='both', color='k', linestyle='dotted', linewidth=0.5, alpha=0.5)
    ax2.set_yticks(y_tick_vals)

    fname = 'images/MPL_' + str(plt_data['N_test']) + '_' + plt_data['network_type'] + '_' + str(
        plt_data['network_parameter']) + '_' + str(plt_data['nx']) + '_' + str(plt_data['rho']) + '_Comp2.pdf'
    try:
        plt.savefig(fname, format='pdf')
        print('File saved as: %s' % (fname))
    except:
        print('Save failed')
    plt.show()

    return None


################################################################

def plot_random_graph_simulation3(plt_data, parameter_list):
    # plt_data - dictionary indexed by test parameter of dictionary of test values
    # Test values: {'A_costs': cost_record_A, 'B_costs': cost_record_B, 'network_parameter': network_parameter, 'nx': nx, 'network_type': network_type, 'N_test': N_test, 'rho': rho, 'alphai': alphai}

    check_values = ['N_test', 'network_type']

    for key1 in plt_data:
        for key2 in plt_data:
            for key3 in check_values:
                if plt_data[key1][key3] != plt_data[key2][key3]:
                    print('ERROR: Data check fails: ', str(plt_data[key1][key3]), ' vs ', str(plt_data[key2][key3]))
                    return None

    fig1 = plt.figure(constrained_layout=True)
    gs1 = GridSpec(2, 1, figure=fig1)

    ax1 = fig1.add_subplot(gs1[0, 0])
    ax2 = fig1.add_subplot(gs1[1, 0], sharex=ax1)

    plot_alpha = [0.8, 0.3, 0.3]
    plot_marker = ['o', 'v', 's']
    plot_line_size = [2, 1, 1]

    y_tick_vals = [0, 0.5, 1]
    nx = [1]

    count = 0

    for p in plt_data:

        if plt_data[p]['nx'] not in nx:
            nx.append(plt_data[p]['nx'])

        A_costs = dc(plt_data[p]['A_costs'])
        B_costs = dc(plt_data[p]['B_costs'])

        A_costs = np.where(np.isnan(A_costs), np.inf, A_costs)
        B_costs = np.where(np.isnan(B_costs), np.inf, B_costs)

        A_values = []
        B_values = []

        A_check_pass = []
        B_check_pass = []

        A_mean = []
        B_mean = []

        A_median = []
        B_median = []

        for i in range(0, plt_data[p]['nx']):
            A_values.append([j for j in A_costs[:, i] if not np.isinf(j)])
            A_check_pass.append((plt_data[p]['N_test'] - np.sum(np.isinf(A_costs[:, i]))) / plt_data[p]['N_test'])
            A_mean.append(np.mean(A_costs[:, i]))
            A_median.append(np.median(A_costs[:, i]))

            B_values.append([j for j in B_costs[:, i] if not np.isinf(j)])
            B_check_pass.append((plt_data[p]['N_test'] - np.sum(np.isinf(B_costs[:, i]))) / plt_data[p]['N_test'])
            B_mean.append(np.mean(B_costs[:, i]))
            B_median.append(np.median(B_costs[:, i]))

        x_val = range(1, 1 + plt_data[p]['nx'])

        ax1.plot(x_val, A_mean, color='C0', marker=plot_marker[count], linewidth=plot_line_size[count],
                 alpha=plot_alpha[count], label='mean(A)')
        ax1.plot(x_val, B_mean, color='C3', marker=plot_marker[count], linewidth=plot_line_size[count],
                 alpha=plot_alpha[count], label='mean(B)')

        ax1.plot(x_val, A_median, color='C0', marker=plot_marker[count], linewidth=plot_line_size[count],
                 alpha=plot_alpha[count], label='median(A)', linestyle='dashed')
        ax1.plot(x_val, B_median, color='C3', marker=plot_marker[count], linewidth=plot_line_size[count],
                 alpha=plot_alpha[count], label='median(B)', linestyle='dashed')

        ax2.plot(x_val, A_check_pass, color='C0', marker=plot_marker[count], linewidth=plot_line_size[count], label='A',
                 alpha=plot_alpha[count])
        ax2.plot(x_val, B_check_pass, color='C3', marker=plot_marker[count], linewidth=plot_line_size[count], label='B',
                 alpha=plot_alpha[count])

        count += 1

    ax1.xaxis.set_tick_params(labelbottom=False)
    ax1.set_yscale('log')
    ax1.set_ylabel('Cost')
    ax1.grid(visible=True, which='both', axis='x', color='k', linestyle='dotted', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel(r'$|S|$')
    ax2.set_ylabel('Control Pass Fraction \n(' + str(plt_data[0]['N_test']) + ' Realizations)')
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=5, integer=True))
    ax2.set_xlim(0.5, max(nx) + 0.5)
    ax2.set_xticks(nx)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks(y_tick_vals)
    ax2.grid(color='k', linestyle='dotted', linewidth=0.5, alpha=0.5)
    ax2.grid(visible=True, which='both', axis='both', color='k', linestyle='dotted', linewidth=0.5, alpha=0.5)

    handle_list1 = [mlines.Line2D([], [], color='C2', linestyle='solid', label='Mean'),
                    mlines.Line2D([], [], color='C2', linestyle='dashed', label='Median')]
    ax1.legend(handles=handle_list1, loc='upper left')

    handle_list2 = []
    for i in range(0, len(parameter_list['p_list'])):
        handle_list2.append(mlines.Line2D([], [], color='C2', marker=plot_marker[i], markersize=10,
                                          label=parameter_list['p_type'] + '=' + str(parameter_list['p_list'][i])))
    handle_list2.append(mpatches.Patch(color='C0', label='A'))
    handle_list2.append(mpatches.Patch(color='C3', label='B'))
    ax2.legend(handles=handle_list2, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)

    fname = 'images/MPL_' + str(plt_data[0]['N_test']) + '_' + plt_data[0]['network_type'] + '_' + str(
        plt_data[0]['rho']) + '_' + parameter_list['p_type'] + '_'
    for i in range(0, len(parameter_list['p_list'])):
        fname += str(parameter_list['p_list'][i])
        if i + 1 < len(parameter_list['p_list']):
            fname += 'vs'
    fname += 'Comp.pdf'

    try:
        plt.savefig(fname, format='pdf')
        print('File saved as: %s' % (fname))
    except:
        print('Save failed')
    plt.show()

    return None


################################################################

def cost_comparison_print(values):
    data_cols = [r"$|S|$", r"$A$", r"$B$", r"$A-B$", r"$\frac{A-B}{A} \times 100$"]
    # data_rows = []

    # for key in values['T_A']['costs']:
    #     data_rows.append(r'$|S|=$ ' + key)
    cost_data = np.zeros((len(values['T_A']['costs']), len(data_cols)))

    for i in range(0, len(values['T_A']['costs'])):
        cost_data[i, 0] = i+1
        cost_data[i, 1] = values['T_A']['costs'][str(i + 1)][-1]
        cost_data[i, 2] = values['T_B']['costs'][str(i + 1)][-1]
        cost_data[i, 3] = cost_data[i, 1] - cost_data[i, 2]
        cost_data[i, 4] = cost_data[i, 3] * 100 / cost_data[i, 1]

    cost_table = pd.DataFrame(cost_data, columns=data_cols)

    cost_table[data_cols[0]] = cost_table[data_cols[0]].astype(int)
    # cost_table[data_cols[0]] = cost_table[data_cols[0]].apply(r'$|S|=$ {:}'.format)
    for key in data_cols[1:-1]:
        cost_table[key] = cost_table[key].apply('{:.2e}'.format)

    cost_table[data_cols[-1]] = cost_table[data_cols[-1]].apply('{:.1f}\%'.format)

    fig1, ax1 = plt.subplots()
    ax1.axis('off')
    ax1.axis('tight')
    table = ax1.table(cellText=cost_table.values, colLabels=cost_table.columns, cellLoc='center', rowLoc='center', loc='center')
    table.set_fontsize(12)
    fig1.tight_layout()

    plt.gcf().canvas.draw()
    points = table.get_window_extent(plt.gcf()._cachedRenderer).get_points()
    points[0, :] -= 10
    points[1, :] += 10
    nbbox = matplotlib.transforms.Bbox.from_extents(points / plt.gcf().dpi)

    try:
        fname = 'images/' + values['file_name'] + '_costcomparison.pdf'
        plt.savefig(fname, format='pdf', bbox_inches=nbbox)
        print('Plot saved as %s' % fname)
    except:
        print('Plot not saved')

    plt.show()

    return None


################################################################


if __name__ == "__main__":
    print('Successfully compiled function file for system cost evaluation')
