import numpy as np
from copy import deepcopy as dc
from functionfile_system_definition import actuator_matrix_to_list, actuator_list_to_matrix

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


def initial_values_init(sys_in=None, T=200, P_max=(10**10)):

    if sys_in is None:
        X0 = None
        alpha_sim = None
        beta_sim = None

    else:
        sys = dc(sys_in)

        if sys['X0'].ndim == 2:
            print('Generating random sample of initial state from given distribution for simulation')
            X0 = np.random.default_rng().multivariate_normal(mean=np.zeros(np.shape(sys['A'])[0]), cov=sys['X0'])
        else:
            X0 = dc(sys['X0'])

        alpha_sim = np.zeros((T, len(sys['alphai'])))
        if np.sum(sys['alphai']) > 0:
            alpha_sim = np.random.default_rng().multivariate_normal(mean=np.zeros(len(sys['alphai'])), cov=np.diag(sys['alphai']), size=T)

        beta_sim = np.zeros((T, len(sys['betaj'])))
        if np.sum(sys['betaj']) > 0:
            beta_sim = np.random.default_rng().multivariate_normal(mean=np.zeros(len(sys['betaj'])), cov=np.diag(sys['betaj']), size=T)

    return_values = {'T': T, 'P_max': P_max, 'alphai': alpha_sim, 'betaj': beta_sim, 'X0': X0}
    # T: max time steps of Riccati iterations
    # P_max: max magnitude of cost matrix eigenvalue before assumed no convergence
    # alphai: simulated state-dependent noise
    # betaj: simulated input-dependent noise
    # X0: initial state simulated if distribution provided
    return return_values


#################################################################

def cost_function_1(sys_in, initial_values=None, feedback=False):
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

    # W_sum = np.zeros_like(sys['W'])
    # W = dc(sys['W'])
    # if np.allclose(W, W_sum):
    #     W_check = False

    while P_check == 1:
        t += 1

        if sys['metric'] == 0:
            J_rec.append(np.trace(P))
        elif sys['metric'] == 1:
            J_rec.append(sys['X0'].T @ P @ sys['X0'])
        elif sys['metric'] == 2:
            J_rec.append(np.trace(P @ sys['X0']))

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

    K = np.zeros((np.shape(B)[1], np.shape(A)[0]))
    if feedback:
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

    # print(B_list)
    # print(S_list)

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
        check_record.append(check_list[idx])
        cost_record.append(cost_list[idx])
        time_record.append(time_list[idx])
        if check_list[idx] != 0:
            idx = np.argmax(time_list)
            check_record[-1] = check_list[idx]
            cost_record[-1] = np.nan
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
    # print(B)
    for i in range(0, np.shape(B)[1]):
        if check[i] == 0:
            ax3.scatter(i*np.ones(np.shape(B)[0]), B[:, i], marker='o', color='C0')
        else:
            ax3.scatter(i*np.ones(np.shape(B)[0]), B[:, i], marker='o', color='C1')
    ax3.invert_yaxis()
    ax3.set_xlabel(r'$|S|$')
    ax3.set_ylabel('Actuated Node')

    if fname is not None:
        fname = 'images/'+fname+'_selection.pdf'
        plt.savefig(fname, format='pdf')

    plt.show()

    return None


################################################################

def simulation_core(sys_in, feedback, initial_values=None):
    sys = dc(sys_in)

    if initial_values is None:
        initial_values = initial_values_init(sys)

    K = feedback['K']

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

        state_trajectory[t + 1, :] = ((dyn_base_mat + dyn_noise) @ state_trajectory[t, :]) #+ w_sim[t, :]
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
        initial_values = initial_values_init(sys_true_in)

        sys_model['X0'] = dc(initial_values['X0'])
        sys_model['metric'] = 1

        sys_true['X0'] = dc(initial_values['X0'])
        sys_true['metric'] = 1

    ret1 = cost_function_1(sys_model, initial_values, feedback=True)
    feedback = {'K': ret1['K']}
    ret2 = simulation_core(sys_true, feedback, initial_values)

    return_values = {'states': ret2['states'], 'costs': ret2['costs'], 'control': ret2['control']}
    return return_values


################################################################

def simulation_actuator_selection(sys_model_in, sys_true_in, initial_values=None):
    sys_model = dc(sys_model_in)
    sys_true = dc(sys_true_in)

    if initial_values is None:
        initial_values = initial_values_init(sys_true_in)

        sys_model['X0'] = dc(initial_values['X0'])
        sys_model['metric'] = 1

        sys_true['X0'] = dc(initial_values['X0'])
        sys_true['metric'] = 1

    if np.shape(sys_model['B']) != np.shape(sys_true['B']):
        print('Controllers are not the same structure')
        return None

    states = {}
    costs = {}
    control = {}

    B_list = actuator_matrix_to_list(sys_model['B'])
    # print(B_list)
    nx = np.shape(sys_model['A'])[0]
    for i in range(0, np.shape(sys_model['B'])[1]):
        sys_model_test = dc(sys_model)
        sys_true_test = dc(sys_true)

        B_test = actuator_list_to_matrix(list(B_list[i:i+1]), nx)
        sys_model_test['B'] = dc(B_test)
        sys_true_test['B'] = dc(B_test)

        if np.sum(sys_model_test['betaj'])>0:
            for j in range(0, len(sys_model_test['betaj'])):
                sys_model_test['Bj'][j, :, :] = dc(B_test)
        if np.sum(sys_true_test['betaj'])>0:
            for j in range(0, len(sys_true_test['betaj'])):
                sys_true_test['Bj'][j, :, :] = dc(B_test)

        test_return = simulation_wrapper(sys_model_test, sys_true_test, initial_values)
        states[str(i+1)] = test_return['states']
        costs[str(i+1)] = test_return['costs']
        control[str(i+1)] = test_return['control']

    return_values = {'label': sys_model['label'], 'states': states, 'costs': costs, 'control': control}
    # states: dictionary of state trajectories for each actuator size
    # costs: dictionary of cost trends for each actuator size
    # control: dictionary of control effort for each actuator size
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
        for i in display_data[s]:
            T = np.shape(display_data[s][i])[0]
            break
    T_range = list(range(0, T))

    if 'states' in display_data:
        ax1 = fig1.add_subplot(gs1[0, 0])
        for i in display_data['states']:
            ax1.plot(T_range, display_data['states'][i], color='C'+i, label=i)
        ax1.set_xlabel(r'$t$')
        ax1.set_ylabel(r'$x_t$')

    if 'costs' in display_data:
        ax2 = fig1.add_subplot(gs1[1, 0])
        for i in display_data['costs']:
            ax2.plot(T_range, display_data['costs'][i], color='C'+i, label=i)
        ax2.set_xlabel(r'$t$')
        ax2.set_ylabel(r'$J^*$')
        ax2.legend(ncol=3)

    if 'control' in display_data:
        ax3 = fig1.add_subplot(gs1[2, 0])
        for i in display_data['control']:
            ax3.plot(T_range, display_data['control'][i], color='C'+i, label=i)
        ax3.set_xlabel(r'$t$')
        ax3.set_ylabel(r'$u_t$')

    if fname is not None:
        fname = 'images/'+fname+'_simulation.pdf'
        plt.savefig(fname, format='pdf')

    plt.show()


################################################################

if __name__ == "__main__":
    print('Successfully compiled function file for system cost evaluation')
