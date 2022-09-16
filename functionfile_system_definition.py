import numpy as np
import networkx as netx
from copy import deepcopy as dc
import pickle

import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
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


def system_package(A_in, B_in=None, alphai_in=None, Ai_in=None, betaj_in=None, Bj_in=None, Q_in=None, R1_in=None, X0_in=None, label_in=None, print_check=True):
    A = dc(A_in)
    nx = np.shape(A)[0]

    if B_in is None:
        B = np.zeros((nx, nx))
        if print_check:
            print('Control input matrix not given - assumed no controller')
    else:
        if np.ndim(B_in) == 1:
            B = actuator_list_to_matrix(B_in, nx)
        elif np.ndim(B_in) == 2:
            B = dc(B_in)
            if np.shape(B)[0] != np.shape(A)[0]:
                if print_check:
                    print('Check control input matrix size')
                return None
            elif np.shape(B)[1] != np.shape(A)[1]:
                B = np.pad(B, ((0, 0), (0, np.shape(A)[1]-np.shape(B)[1])), 'constant')
        else:
            if print_check:
                print('Check control input matrix')
            return None
    nu = np.shape(B)[1]

    if label_in is None:
        label = 'System'
    else:
        label = dc(label_in)

    if alphai_in is None:
        alphai = [0]
        Ai = np.expand_dims(np.zeros_like(A), axis=0)
        if print_check:
            print('Actuator noise matrix not specified - assumed 0')
    else:
        alphai = dc(alphai_in)
        if Ai_in is None:
            Ai = np.expand_dims(A, axis=0)
        else:
            if np.ndim(Ai_in) == 2:
                Ai = np.expand_dims(Ai_in, axis=0)
            elif np.ndim(Ai_in) == 3:
                Ai = dc(Ai_in)
            else:
                if print_check:
                    print('Check actuator noise matrix')
                return None

    if betaj_in is None:
        betaj = [0]
        Bj = np.expand_dims(np.zeros_like(B), axis=0)
        if print_check:
            print('Control noise matrix not specified - assumed 0')
    else:
        betaj = dc(betaj_in)
        if Bj_in is None:
            Bj = np.expand_dims(B, axis=0)
        else:
            if np.ndim(Bj_in) == 2:
                Bj = np.expand_dims(Bj_in, axis=0)
            elif np.ndim(Bj_in) == 3:
                Bj = dc(Bj_in)
            else:
                if print_check:
                    print('Check control noise matrix')
                return None

    if Q_in is None:
        Q = np.identity(nx)
    else:
        Q = dc(Q_in)

    if R1_in is None:
        R1 = np.identity(nu)
    else:
        R1 = dc(R1_in)

    if X0_in is None:
        if print_check:
            print('No initial state specified')
        X0 = np.zeros(nx)
        metric = 0
    elif np.ndim(X0_in) == 1:
        if print_check:
            print('Initial state vector specified')
        X0 = dc(X0_in)
        metric = 1
    elif np.ndim(X0_in) == 2:
        if print_check:
            print('Initial state distribution specified')
        X0 = dc(X0_in)
        metric = 2
    else:
        if print_check:
            print('Check initial state distribution')
        return None

    sys = {'label': label, 'A': A, 'B': B, 'alphai': alphai, 'Ai': Ai, 'betaj': betaj, 'Bj': Bj, 'Q': Q, 'R1': R1, 'X0': X0, 'metric': metric}
    return sys


#######################################################

def system_check(sys_in=None, print_message=False):
    # Dimension check for system dynamics and cost parameters
    # Return dictionary with ['check']=1 if dimensions are OKAY, 0 otherwise
    # Print success check if print_message is True

    if sys_in is None:
        print('No system given')
        return None

    sys = dc(sys_in)

    check = 1

    nx = np.shape(sys['A'])[0]
    nu = np.shape(sys['B'])[1]

    if np.shape(sys['Q']) != (nx, nx):
        print('Incorrect Q dimension - expected:(', nx, ', ', nx, ') | got:', np.shape(sys['Q']))
        check = 0

    if np.shape(sys['R1']) != (nu, nu):
        print('Incorrect R1 dimension - expected:(', nu, ', ', nu, ') | got:', np.shape(sys['R1']))
        check = 0

    n_alpha = len(sys['alphai'])
    n_beta = len(sys['betaj'])

    if np.shape(sys['Ai']) != (n_alpha, nx, nx):
        print('Incorrect Ai dimensions - expected:(', n_alpha, ',', nx, ', ', nx, ') | got:', np.shape(sys['Ai']))
        check = 0

    if np.shape(sys['Bj']) != (n_beta, nu, nu):
        print('Incorrect Bj dimensions - expected:(', n_beta, ',', nu, ', ', nu, ') | got:', np.shape(sys['Bj']))
        check = 0

    if np.ndim(sys['X0']) == 1 and np.shape(sys['X0']) != (nx,):
        print('Incorrect X0 - expected:(', nx, ',', ') | got:', np.shape(sys['X0']))
        check = 0
    elif np.ndim(sys['X0']) == 2 and np.shape(sys['X0']) != (nx, nx):
        print('Incorrect X0 - expected:(', nx, ',', nx, ') | got:', np.shape(sys['X0']))
        check = 0

    if not check:
        print('System error')
    elif print_message:
        print('System check - OKAY')

    return_value = {'check': check}
    return return_value

#######################################################

def actuator_list_to_matrix(B_list, nx):
    B = np.zeros((nx, nx))
    for i in range(0, len(B_list)):
        B[B_list[i], i] = 1

    return B


#######################################################

def actuator_matrix_to_list(B_in):
    B_list = []
    B = dc(B_in)

    for i in range(0, np.shape(B)[1]):
        idx = np.argmax(B[:, i])
        if B[idx, i]!=0:
            B_list.append(idx)

    return B_list


#######################################################

def actuator_random_selection(nx_in, nu_in):
    nx = dc(nx_in)
    nu = dc(nu_in)

    B_list = np.random.default_rng().choice(range(0, nx), nu, replace=False)
    B = actuator_list_to_matrix(B_list, nx)

    return_values = {'B': B, 'B_list': B_list}
    return return_values


#######################################################

def sys_to_file(sys_in, f_name=None):
    sys = dc(sys_in)

    if f_name is None:
        f_name = sys['label']

    f_name = 'system_model/' + f_name + '.pickle'
    try:
        f_open = open(f_name, 'wb')
    except:
        print('File not writable\n')
        return None

    pickle.dump(sys, f_open)
    f_open.close()
    print('System saved to file @', f_name, '\n')
    return None


#######################################################

def sys_from_file(f_name='sys_model'):
    f_name = 'system_model/' + f_name + '.pickle'
    try:
        f_open = open(f_name, 'rb')
    except:
        print('File not readable\n')
        return None

    sys = pickle.load(f_open)
    f_open.close()
    print('System read from file @', f_name, '\n')
    return sys


#######################################################

def system_display_matrix(sys_in, fname=None):
    sys = dc(sys_in)

    nx = np.shape(sys['A'])[0]
    nu = np.shape(sys['B'])[1]
    # nv = np.shape(sys['F'])[1]

    fig1 = plt.figure(constrained_layout=True)
    gs1 = GridSpec(2, 4, figure=fig1)

    # cm = plt.get_cmap('Blues')

    ax1 = fig1.add_subplot(gs1[0, 0])
    a1 = ax1.imshow(sys['A'], extent=[0.5, nx + 0.5, nx + 0.5, 0.5])
    ax1.set_title(r'$A$')
    # plt.colorbar(a1, ax=ax1, location='bottom')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2, min_n_ticks=3))
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2, min_n_ticks=3))

    if np.sum(sys['B']) > 0:
        ax2 = fig1.add_subplot(gs1[0, 1])
        a2 = ax2.imshow(sys['B'], extent=[0.5, nu + 0.5, nx + 0.5, 0.5])
        ax2.set_title(r'$B$')
        # plt.colorbar(a2, ax=ax2, location='bottom')
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2, min_n_ticks=3))
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2, min_n_ticks=3))

    net_alpha = np.sum(sys['alphai'])
    net_beta = np.sum(sys['betaj'])
    if net_alpha + net_beta > 0:
        ax4 = fig1.add_subplot(gs1[1, 0])
        ax4_col = 0

        if net_alpha != 0:
            a4 = ax4.scatter(range(1, len(sys['alphai']) + 1), sys['alphai'], c='C0', marker='1', alpha=0.8, label=r'$\alpha_i$')
            ax4_col += 1
            Ai_net = np.zeros_like(sys['A'])
            for i in range(0, len(sys['alphai'])):
                Ai_net += sys['Ai'][i, :, :]*sys['alphai'][i] #/ net_alpha
            ax5 = fig1.add_subplot(gs1[1, 1])
            a5 = ax5.imshow(Ai_net, extent=[0.5, nx + 0.5, nx + 0.5, 0.5])
            # plt.colorbar(a5, ax=ax5, location='bottom')
            ax5.set_title(r'$\sum \alpha_i A_i$')
            ax5.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2, min_n_ticks=3))
            ax5.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2, min_n_ticks=3))

        if net_beta != 0:
            a4 = ax4.scatter(range(1, len(sys['betaj']) + 1), sys['betaj'], c='C2', marker='2', alpha=0.8, label=r'$\beta_j$')
            ax4_col += 1
            Bj_net = np.zeros_like(sys['B'])
            for j in range(0, len(sys['betaj'])):
                Bj_net += sys['Bj'][j, :, :]*sys['betaj'][j] #/ net_beta
            if net_alpha != 0:
                ax6 = fig1.add_subplot(gs1[1, 2])
            else:
                ax6 = fig1.add_subplot(gs1[1, 1])
            a6 = ax6.imshow(Bj_net, extent=[0.5, nu + 0.5, nx + 0.5, 0.5])
            # plt.colorbar(a6, ax=ax6, location='bottom')
            ax6.set_title(r'$\sum \beta_j B_j$')
            ax6.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2, min_n_ticks=3))
            ax6.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2, min_n_ticks=3))

        ax4.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2, min_n_ticks=3))
        ax4.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2, min_n_ticks=3))
        ax4.legend(markerfirst=False, framealpha=0.2, handlelength=1, labelspacing=0.4, columnspacing=0.5, ncol=ax4_col)
        ax4.set_title('MPL Covariances')

    # ax10 = fig1.add_subplot(gs1[0, 3])
    # a10 = ax10.imshow(sys['Q'], extent=[0.5, nx + 0.5, nx + 0.5, 0.5])
    # ax10.set_title(r'$Q$')
    # plt.colorbar(a10, ax=ax10, location='bottom')
    # ax10.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2, min_n_ticks=3))
    # ax10.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2, min_n_ticks=3))
    #
    # ax11 = fig1.add_subplot(gs1[1, 3])
    # a11 = ax11.imshow(sys['R1'], extent=[0.5, nu + 0.5, nu + 0.5, 0.5])
    # ax11.set_title(r'$R_1$')
    # plt.colorbar(a11, ax=ax11, location='bottom')
    # ax11.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2, min_n_ticks=3))
    # ax11.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2, min_n_ticks=3))

    # plt.suptitle(sys['label'])

    if fname is None:
        fname = 'images/' + sys['label'] + '.pdf'
    else:
        fname = 'images/' + fname + '_' + sys['label'] + '.pdf'

    try:
        plt.savefig(fname, format='pdf')
        print('Image save @', fname)
    except:
        print('Inaccessible save name/directory')
    plt.show()

    return None


#######################################################

def create_graph(nx_in, type='cycle', p=None, self_loop=True):
    nx = dc(nx_in)
    net_check = True

    G = None
    while net_check:
        if type == 'cycle':
            G = netx.generators.classic.cycle_graph(nx)
        elif type == 'path':
            G = netx.generators.classic.path_graph(nx)
        elif type == 'ER':
            if p is None:
                print('Specify edge probability for ER-graph')
                return None
            else:
                G = netx.generators.random_graphs.erdos_renyi_graph(nx, p)
        elif type == 'BA':
            if p is None:
                print('Specify initial network size for BA-graph')
                return None
            else:
                G = netx.generators.random_graphs.barabasi_albert_graph(nx, p)
        else:
            print('Check network type')
            return None
        if netx.algorithms.components.is_connected(G):
            net_check = False

    if G is None:
        print('Error: Check graph generator')
        return None

    Adj = netx.to_numpy_array(G)
    if self_loop:
        Adj += np.identity(nx)

    e = np.max(np.abs(np.linalg.eigvals(Adj)))
    A = Adj/e

    return_values = {'A': A, 'eig_max': e, 'Adj': Adj}
    return return_values


#####################################################

def matrix_splitter(A):

    A_split = np.expand_dims(np.zeros_like(A), axis=0)

    for i in range(0, np.shape(A)[0]):
        for j in range(0, np.shape(A)[1]):
            if A[i, j] != 0:
                # print(np.shape(A_split))
                A_add = np.zeros_like(A)
                A_add[i, j] = 1
                # print(A_add)
                A_split = np.append(A_split, np.expand_dims(A_add, axis=0), axis=0)

    # print(np.shape(A_split))
    A_split = A_split[1:, :, :]
    # print(np.shape(A_split))

    return A_split

#####################################################


if __name__ == "__main__":

    print('Successfully compiled function file for system definition')
