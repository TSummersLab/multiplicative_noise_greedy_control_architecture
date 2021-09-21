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


def system_package(A_in, B_in=None, alphai_in=None, Ai_in=None, betaj_in=None, Bj_in=None, Q_in=None, R1_in=None, label_in=None):
    A = dc(A_in)
    nx = np.shape(A)[0]

    if B_in is None:
        B = np.zeros((nx, nx))
        print('Control input matrix not given')
    else:
        if np.ndim(B_in) == 1:
            B = actuator_matrix(B_in, nx)
        elif np.ndim(B_in) == 2:
            B = dc(B_in)
        else:
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
        print('Actuator noise matrix not specified')
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
                print('Check actuator noise matrix')

    if betaj_in is None:
        betaj = [0]
        Bj = np.expand_dims(np.zeros_like(B), axis=0)
        print('Control noise matrix not specified')
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
                print('Check control noise matrix')

    if Q_in is None:
        Q = np.identity(nx)
    else:
        Q = dc(Q_in)

    if R1_in is None:
        R1 = np.identity(nu)
    else:
        R1 = dc(R1_in)

    sys = {'A': A, 'B': B, 'label': label}
    return sys


#######################################################

def actuator_matrix(B_in, nx):
    B = np.zeros((nx, nx))
    for i in range(0, len(B_in)):
        B[B_in[i], i] = 1

    return B


#######################################################

def sys_to_file(sys_in, f_name='sys_model'):
    sys = dc(sys_in)
    f_name = 'system_model/' + f_name + '.pickle'
    try:
        f_open = open(f_name, 'wb')
    except:
        print('File not writable')
        return

    pickle.dump(sys, f_open)
    f_open.close()
    print('System saved to file @', f_name)
    return


#######################################################

def sys_from_file(f_name='sys_model'):
    f_name = 'system_model/' + f_name + '.pickle'
    try:
        f_open = open(f_name, 'rb')
    except:
        print('File not readable')
        return

    sys = pickle.load(f_open)
    f_open.close()
    print('System read from file @', f_name)
    return sys


#######################################################

def system_display_matrix(sys_in, fname=None, imgtitle=None):
    sys = dc(sys_in)

    nx = np.shape(sys['A'])[0]
    nu = np.shape(sys['B'])[1]
    nv = np.shape(sys['F'])[1]

    fig1 = plt.figure(constrained_layout=True)
    gs1 = GridSpec(3, 4, figure=fig1)

    # cm = plt.get_cmap('Blues')

    ax1 = fig1.add_subplot(gs1[0, 0])
    a1 = ax1.imshow(sys['A'], extent=[0.5, nx + 0.5, nx + 0.5, 0.5])
    ax1.set_title(r'$A$')
    plt.colorbar(a1, ax=ax1, location='bottom')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))

    ax2 = fig1.add_subplot(gs1[1, 0])
    a2 = ax2.imshow(actuator_matrix(sys['B'], nx), extent=[0.5, nu + 0.5, nx + 0.5, 0.5])
    ax2.set_title(r'$B$')
    plt.colorbar(a2, ax=ax2, location='bottom')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))

    ax3 = fig1.add_subplot(gs1[2, 0])
    a3 = ax3.imshow(actuator_matrix(sys['F'], nx), extent=[0.5, nv + 0.5, nx + 0.5, 0.5])
    ax3.set_title(r'$F$')
    plt.colorbar(a3, ax=ax3, location='bottom')
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))

    net_alpha = np.sum(sys['alphai'])
    net_beta = np.sum(sys['betaj'])
    net_gamma = np.sum(sys['gammak'])
    if net_alpha+net_beta+net_gamma > 0:
        ax4 = fig1.add_subplot(gs1[0, 1])
        ax4_col = 0

        if net_alpha != 0:
            a4 = ax4.scatter(range(1, len(sys['alphai']) + 1), sys['alphai'], c='C0', marker='1', alpha=0.8, label=r'$\alpha_i$')
            ax4_col += 1
            Ai_net = np.zeros_like(sys['A'])
            for i in range(0, len(sys['alphai'])):
                Ai_net += sys['Ai'][i, :, :]*sys['alphai'][i] #/ net_alpha
            ax5 = fig1.add_subplot(gs1[0, 2])
            a5 = ax5.imshow(Ai_net, extent=[0.5, nx + 0.5, nx + 0.5, 0.5])
            plt.colorbar(a5, ax=ax5, location='bottom')
            ax5.set_title(r'$\sum \alpha_i A_i$')
            ax5.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
            ax5.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))

        if net_beta != 0:
            a4 = ax4.scatter(range(1, len(sys['betaj']) + 1), sys['betaj'], c='C2', marker='2', alpha=0.8, label=r'$\beta_j$')
            ax4_col += 1
            Bj_net = np.zeros_like(sys['B'])
            for j in range(0, len(sys['betaj'])):
                Bj_net += sys['Bj'][j, :, :]*sys['betaj'][j] #/ net_beta
            ax6 = fig1.add_subplot(gs1[1, 2])
            a6 = ax6.imshow(Bj_net, extent=[0.5, nu + 0.5, nx + 0.5, 0.5])
            plt.colorbar(a6, ax=ax6, location='bottom')
            ax6.set_title(r'$\sum \beta_j B_j$')
            ax6.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
            ax6.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))

        if net_gamma != 0:
            a4 = ax4.scatter(range(1, len(sys['gammak']) + 1), sys['gammak'], c='C1', marker='3', alpha=0.8, label=r'$\gamma_k$')
            ax4_col += 1
            Fk_net = np.zeros_like(sys['F'])
            for k in range(0, len(sys['gammak'])):
                Fk_net += sys['Fk'][k, :, :]*sys['gammak'][k] #/ net_gamma
            ax7 = fig1.add_subplot(gs1[2, 2])
            a7 = ax7.imshow(Fk_net, extent=[0.5, nv + 0.5, nx + 0.5, 0.5])
            plt.colorbar(a7, ax=ax7, location='bottom')
            ax7.set_title(r'$\sum \gamma_k F_k$')
            ax7.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
            ax7.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))

        ax4.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=max(len(sys['alphai']), len(sys['betaj']), len(sys['gammak']))))
        ax4.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
        ax4.legend(markerfirst=False, framealpha=0.2, handlelength=1, labelspacing=0.4, columnspacing=0.5, ncol=ax4_col)
        ax4.set_title('MPL Covariances')

    ax10 = fig1.add_subplot(gs1[0, 3])
    a10 = ax10.imshow(sys['Q'], extent=[0.5, nx + 0.5, nx + 0.5, 0.5])
    ax10.set_title(r'$Q$')
    plt.colorbar(a10, ax=ax10, location='bottom')
    ax10.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
    ax10.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))

    ax11 = fig1.add_subplot(gs1[1, 3])
    a11 = ax11.imshow(sys['R1'], extent=[0.5, nu + 0.5, nu + 0.5, 0.5])
    ax11.set_title(r'$R_1$')
    plt.colorbar(a11, ax=ax11, location='bottom')
    ax11.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
    ax11.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))

    ax12 = fig1.add_subplot(gs1[2, 3])
    a12 = ax12.imshow(sys['R2'], extent=[0.5, nv + 0.5, nv + 0.5, 0.5])
    ax12.set_title(r'$R_2$')
    plt.colorbar(a12, ax=ax12, location='bottom')
    ax12.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
    ax12.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))

    if not np.allclose(sys['W'], np.zeros_like(sys['W'])):
        ax13 = fig1.add_subplot(gs1[1, 1])
        a13 = ax13.imshow(sys['W'], extent=[0.5, nv + 0.5, nv + 0.5, 0.5])
        ax13.set_title(r'$W$')
        plt.colorbar(a13, ax=ax13, location='bottom')
        ax13.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
        ax13.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))

    if imgtitle is not None:
        fig1.suptitle(imgtitle)
    elif sys['label'] is not None:
        fig1.suptitle(sys['label'])

    if fname is not None:
        try:
            plt.savefig(fname+'.eps', format='eps')
        except:
            print('Incorrect save name/directory')
    plt.show()

    return None


#######################################################
