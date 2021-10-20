import numpy as np
from copy import deepcopy as dc
from functionfile_system_definition import actuator_matrix_to_list, actuator_list_to_matrix, system_check

import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
# from matplotlib.ticker import MaxNLocator

matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8
# matplotlib.rcParams['image.cmap'] = 'Blues'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True


def initial_values_init(sys_in=None, T=200, P_max=(10**10), P_min=(10**(-8))):
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
            alpha_sim = np.random.default_rng().multivariate_normal(mean=np.zeros(len(sys['alphai'])), cov=np.diag(sys['alphai']), size=T)
        return_values['alphai'] = alpha_sim

        beta_sim = np.zeros((T, len(sys['betaj'])))
        if np.sum(sys['betaj']) > 0:
            beta_sim = np.random.default_rng().multivariate_normal(mean=np.zeros(len(sys['betaj'])), cov=np.diag(sys['betaj']), size=T)
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
            if np.sum(sys_test['betaj'])>0:
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

    return_values = {'system': sys_return, 'cost_trend': cost_record, 'time_trend': time_record, 'check_trend': check_record}
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
    sc1 = np.nan*np.ones(len(check))
    sc2 = np.nan*np.ones(len(check))
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
                B[j, i:np.shape(B)[1]] = (j+1)*np.ones(np.shape(B)[1]-i)
    # B = np.pad(B, ((0,0),(1,0)), 'constant')
    print(B)
    for i in range(0, np.shape(B)[1]):
        if check[i] == 0:
            ax3.scatter(i*np.ones(np.shape(B)[0]), B[:, i], marker='o', color='C0')
        else:
            ax3.scatter(i*np.ones(np.shape(B)[0]), B[:, i], marker='o', color='C1')
    ax3.invert_yaxis()
    ax3.set_xlabel(r'$|S|$')
    ax3.set_ylabel('Actuated Node')

    if fname is not None:
        fig1.suptitle(fname)
        fname = 'images/'+fname+'_selection.pdf'
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
            return_values[str(i)] = {'check': test_ret['P_check'], 'cost': test_ret['J_trend'][-1], 'time': test_ret['t']}

    # return dictionary: key is number of actuators
    return return_values


################################################################

def plot_actuator_selection_2(values, fname=None):
    fig1 = plt.figure(constrained_layout=True)
    gs1 = GridSpec(3, 1, figure=fig1)

    n_vals = len(values)-2
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
                B[j, i:np.shape(B)[1]] = (j+1)*np.ones(np.shape(B)[1]-i)
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
    fname = 'images/'+fname+'_selection.pdf'
    plt.savefig(fname, format='pdf')
    plt.show()

    return None


################################################################

def plot_actuator_selection_comparison_1(values1, values2, fname=None):
    fig1 = plt.figure(constrained_layout=True)
    gs1 = GridSpec(3, 1, figure=fig1)

    n_vals = len(values1)-2
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
                B1[j, i:np.shape(B1)[1]] = (j+1)*np.ones(np.shape(B1)[1]-i)
            if B2[j, i] == 0:
                B2[j, i] = np.nan
            else:
                B2[j, i:np.shape(B2)[1]] = (j+1)*np.ones(np.shape(B2)[1]-i)
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
    fname = 'images/'+fname+'_selection.pdf'
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

    state_trajectory = np.nan * np.ones((T_sim+1, nx))
    state_trajectory[0, :] = X0

    cost_trajectory = np.nan * np.ones((T_sim+1))
    cost_trajectory[0] = X0.T @ Q @ X0

    control_effort = np.nan * np.ones((T_sim + 1, nu))

    cost_mat = Q + (K.T @ R1 @ K)
    dyn_base_mat = A + (B @ K)

    for t in range(0, T_sim):

        cost_trajectory[t + 1] = cost_trajectory[t] + (state_trajectory[t].T @ cost_mat @ state_trajectory[t])

        dyn_noise = np.zeros((nx, nx))
        if np.sum(alpha_sim[t, :]) > 0:
            for i in range(0, np.shape(alpha_sim)[1]):
                dyn_noise += alpha_sim[t, i] * Ai[i, :, :]
        if np.sum(beta_sim[t, :]) > 0:
            for j in range(0, np.shape(beta_sim)[1]):
                dyn_noise += beta_sim[t, j] * (Bj[j, :, :] @ K)

        state_trajectory[t + 1, :] = ((dyn_base_mat + dyn_noise) @ state_trajectory[t, :])
        control_effort[t, :] = K @ state_trajectory[t, :]

        if np.abs(cost_trajectory[t + 1]) > initial_values['P_max']:
            print('====> Breaking current simulation at t=', t, ' as cumulative cost magnitude exceed 10^(10)')
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

    return_values = {'label': sys_model['label'] + ' ' + sys_true['label'], 'states': states, 'costs': costs, 'control': control}
    # states: dictionary of state trajectories for each actuator size
    # costs: dictionary of cost trends for each actuator size
    # control: dictionary of control effort for each actuator size
    return return_values


################################################################

def simulation_nom_vs_mpl(sys_nom_in, sys_mpl_in, sys_true_in, initial_values=None):
    sys_true = dc(sys_true_in)
    sys_mpl = dc(sys_mpl_in)
    sys_nom = dc(sys_nom_in)

    if initial_values is None:
        initial_values = initial_values_init(sys_true)

    sys_mpl = dc(actuator_selection_cost_1(sys_mpl, initial_values=initial_values)['system'])
    sys_nom = dc(actuator_selection_cost_1(sys_nom, initial_values=initial_values)['system'])

    ret_nom = simulation_actuator_selection(sys_nom, sys_true, initial_values=initial_values)
    ret_mpl = simulation_actuator_selection(sys_mpl, sys_true, initial_values=initial_values)

    ret_nom['label'] = sys_true['label'] + 'from Nominal'
    ret_mpl['label'] = sys_mpl['label'] + 'from MPL'

    return_values = {'T_Nom': ret_nom, 'T_MPL': ret_mpl, 'system_mpl': sys_mpl, 'system_nom': sys_nom}
    return return_values


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
            ax1.plot(T_range, display_data['states'][i], color='C'+i, alpha=0.5, label=i)
        # ax1.set_yscale('log')
        ax1.set_xlabel(r'$t$')
        ax1.set_ylabel(r'$x_t$')

    if 'costs' in display_data:
        ax2 = fig1.add_subplot(gs1[1, 0])
        for i in display_data['costs']:
            ax2.plot(T_range, display_data['costs'][i], color='C'+i, alpha=0.5, label=i)
        ax2.set_xlabel(r'$t$')
        ax2.set_ylabel(r'$J^*$')
        ax2.set_yscale('log')
        ax2.legend(ncol=3)

    if 'control' in display_data:
        ax3 = fig1.add_subplot(gs1[2, 0])
        for i in display_data['control']:
            ax3.plot(T_range, display_data['control'][i], color='C'+i, alpha=0.5, label=i)
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

def plot_simulation_nom_vs_mpl_1(values, fname=None):

    Nom_values = dc(values['T_Nom'])
    MPL_values = dc(values['T_MPL'])

    fig1 = plt.figure(constrained_layout=True)
    gs1 = GridSpec(3, 1, figure=fig1)

    T = np.shape(Nom_values['states']['1'])[0]
    T_range = list(range(0, T))

    ax1 = fig1.add_subplot(gs1[0, 0])
    for key in Nom_values['states']:
        # ax1.plot(T_range, Nom_values['states'][key], marker='o', alpha=0.5, color='C'+key)
        # ax1.plot(T_range, MPL_values['states'][key], marker='x', alpha=0.5, color='C'+key)
        ax1.plot(T_range, Nom_values['states'][key], ls=':', alpha=0.5, color='C' + key)
        ax1.plot(T_range, MPL_values['states'][key], ls='-.', alpha=0.5, color='C' + key)
    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$x_t$')

    ax2 = fig1.add_subplot(gs1[1, 0], sharex=ax1)
    for key in Nom_values['control']:
        # ax2.plot(T_range, Nom_values['control'][key], marker='o', alpha=0.5, color='C'+key)
        # ax2.plot(T_range, MPL_values['control'][key], marker='x', alpha=0.5, color='C'+key)
        ax2.plot(T_range, Nom_values['control'][key], ls=':', alpha=0.5, color='C' + key)
        ax2.plot(T_range, MPL_values['control'][key], ls='-.', alpha=0.5, color='C' + key)
    ax2.set_xlabel(r'$t$')
    ax2.set_ylabel(r'$u_t$')

    ax3 = fig1.add_subplot(gs1[2, 0], sharex=ax1)
    for key in Nom_values['costs']:
        # ax3.plot(T_range, Nom_values['costs'][key], marker='o', markeredgewidth=0.5, alpha=0.5, color='C'+key)
        # ax3.plot(T_range, MPL_values['costs'][key], marker='x', markeredgewidth=0.5, alpha=0.5, color='C'+key)
        ax3.plot(T_range, Nom_values['costs'][key], ls=':', alpha=0.5, color='C' + key)
        ax3.plot(T_range, MPL_values['costs'][key], ls='-.', alpha=0.5, color='C' + key)
        # ax3.plot(T_range, Nom_values['costs'][key], ls=':', marker='x', alpha=0.5, color='C' + key)
        # ax3.plot(T_range, MPL_values['costs'][key], alpha=0.5, color='C' + key)
    ax3.set_xlabel(r'$t$')
    ax3.set_ylabel(r'$J_t$')
    ax3.set_yscale('log')

    plt.show()

    return None


################################################################

def plot_simulation_nom_vs_mpl_2(values, fname=None):
    Nom_values = dc(values['T_Nom'])
    MPL_values = dc(values['T_MPL'])

    fig1 = plt.figure(constrained_layout=True)
    gs1 = GridSpec(2, 1, figure=fig1)

    T = np.shape(Nom_values['costs']['1'])[0]
    T_range = list(range(0, T))

    ax1 = fig1.add_subplot(gs1[0, 0])
    for key in Nom_values['costs']:
        ax1.plot(T_range, Nom_values['costs'][key], ls=':', alpha=0.5, color='C' + key)
        ax1.plot(T_range, MPL_values['costs'][key], ls='-.', alpha=0.5, color='C' + key)
        # ax1.plot(T_range, Nom_values['costs'][key], marker='o', alpha=0.5, color='C' + key)
        # ax1.plot(T_range, MPL_values['costs'][key], marker='x', alpha=0.5, color='C' + key)
        # ax1.plot(T_range, Nom_values['costs'][key], ls=':', marker='x', alpha=0.5, color='C' + key)
        # ax1.plot(T_range, MPL_values['costs'][key], alpha=0.5, color='C' + key)
    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$J_t$')
    ax1.set_yscale('log')

    ax2 = fig1.add_subplot(gs1[1, 0], sharex=ax1)
    for key in Nom_values['costs']:
        ax2.plot(T_range, Nom_values['costs'][key]-MPL_values['costs'][key], alpha=0.7, color='C'+key, label=key)
    ax2.set_xlabel(r'$t$')
    ax2.set_ylabel(r'$J_t$ (Nominal - MPL feedback)')
    ax2.legend(ncol=3)

    plt.show()

    return None


################################################################

def actuator_comparison(sys1_in, sys2_in, figplt=True):
    S1 = dc(sys1_in)
    S2 = dc(sys2_in)

    return_value = {}

    if np.allclose(S1['B'], S2['B']):
        print('Both control sets are close/equal')
        return_value['act_comp'] = 0
        if figplt:
            fig1 = plt.figure(constrained_layout=True)
            gs1 = GridSpec(1, 1, figure=fig1)
            ax1 = fig1.add_subplot(gs1[0, 0])
            B = dc(S1['B'])
            for i in range(0, np.shape(B)[1]):
                for j in range(0, np.shape(B)[0]):
                    if B[j, i] == 0:
                        B[j, i] = np.nan
                    else:
                        B[j, i:np.shape(B)[1]] = (j + 1) * np.ones(np.shape(B)[1] - i)
                ax1.scatter((i+1) * np.ones(np.shape(B)[0]), B[:, i], marker='o', color='C0', s=100)
            # ax1.set_ylim(bottom=1-0.5, top=np.shape(B)[0]+0.5)
            # ax1.set_xlim(left=1-0.5, right=np.shape(B)[1]+0.5)
            ax1.invert_yaxis()
            ax1.set_xlabel(r'$|S|$')
            ax1.set_ylabel('Node number')
            ax1.set_title('Actuator Set (B) comparison')
            plt.show()
        else:
            print(S1['label'], ' B = ', S2['label'], ' B:\n', S2['B'])

    else:
        print('Control sets are different')
        return_value['act_comp'] = 1
        if figplt:
            fig1 = plt.figure(constrained_layout=True)
            gs1 = GridSpec(1, 1, figure=fig1)
            ax1 = fig1.add_subplot(gs1[0, 0])
            B1 = S1['B']
            B2 = S2['B']
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
                ax1.scatter((i+1) * np.ones(np.shape(B1)[0]), B1[:, i], marker='1', color='C0', s=100)
                ax1.scatter((i+1) * np.ones(np.shape(B2)[0]), B2[:, i], marker='2', color='C1', s=100)
            # ax1.set_ylim(bottom=1-0.5, top=np.shape(B1)[0]+0.5)
            # ax1.set_xlim(left=1-0.5, right=np.shape(B1)[1]+0.5)
            ax1.invert_yaxis()
            ax1.set_xlabel(r'$|S|$')
            ax1.set_ylabel('Node number')
            ax1.set_title('Actuator Set (B) comparison')
            ax1.legend([S1['label'], S2['label']], framealpha=0.5)
            plt.show()
        else:
            print(S1['label'], ' B:\n', S1['B'])
            print(S2['label'], ' B:\n', S2['B'])
            print('B diff (%s - %s):' % (S1['label'], S2['label']))
            print(S1['B'] - S2['B'])


    # print(S1['label'], ' B:\n', S1['B'])
    # print(S2['label'], ' B:\n', S2['B'])

    return return_value


################################################################


if __name__ == "__main__":
    print('Successfully compiled function file for system cost evaluation')
