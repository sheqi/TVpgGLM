# Plot simulation results of tv_model
import sys
sys.path.append("/Users/roger/Dropbox/TVpgGLM-v1/TVpgGLM/utils")

import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import xcorr
from plot_networks import draw_curvy_network
import seaborn as sns
from hips.plotting.colormaps import harvard_colors

sns.set_style("white")
paper_rc = {'lines.linewidth': 1, 'lines.markersize': 10,
            'font.size': 15, 'axes.labelsize':15, 'axes.titlesize':15,
            'xtick.labelsize': 15, 'ytick.labelsize': 15}
sns.set_context("paper", rc = paper_rc)
plt.ion()

color = harvard_colors()[0:10]

with open('TVpgGLM/results/sythetic_tv_N10.pickle', 'rb') as f:
    fr_true, w_true, fr_est, fr_std, w_est, Y= pickle.load(f)

N = 10
N_smpls = 50

########################################################
##Illustration: layout for true/estimated network (p1)##
########################################################

G = nx.MultiDiGraph([(x,y) for x in range(N) for y in range(N)])
pos = nx.circular_layout(G)

# True weights extration
def _circular_layout(w, t, ax):
    w1 = w
    t1 = t
    ax1 = ax
    dT = 200  # interval of weights
    w_true_smpls = w1[:,np.arange(t1,1000,dT),:,0] # [time, out, in]
    edge_width = 50*w_true_smpls[0,:,:].reshape(10*10,)

    edge_color = np.zeros((100,3))
    edge_color[np.where(edge_width>0)] = color[1]
    edge_color[np.where(edge_width<0)] = color[0]

    return draw_curvy_network(G, pos, ax=ax1, node_color='k',
                       node_edge_color='k', edge_width=edge_width,
                       edge_color=edge_color)

for t in range(5):
    plt.figure()
    ax = plt.gca()
    _circular_layout(w=w_est[N_smpls//2:].mean(0), t=t, ax=ax)
    #_circular_layout(w=w_true, t=t, ax=ax)
    ax.autoscale()
    plt.axis('equal')
    plt.axis('off')
    plt.title('t='+str(200*(t+1)))
    plt.tight_layout()
    plt.savefig("TVpgGLM/fig/syn_tv_N10_T"+str(200*(t+1))+".png",dpi=500)

##############################
##Raster plot and rates (p2)##
##############################
pltslice=slice(0, 500)
fig,axs = plt.subplots(2,2)
k = 0
for i in range(2):
    for j in range(2):
        tn = np.where(Y[pltslice, k])[0]
        axs[i,j].plot(tn, np.ones_like(tn), 'ko', markersize=4)
        axs[i,j].plot(fr_est[pltslice,k],color=color[0])
        axs[i,j].plot(fr_true[pltslice, k],color=color[1])
        # sausage_plot(np.arange(pltslice.start, pltslice.stop),
        #             fr_est[pltslice,n],
        #             #fr_true[pltslice, n],
        #             3*fr_std[pltslice,n],
        #             sgax=axs[n],
        #             alpha=0.5)
        axs[i,j].set_ylim(-0.05, 1.1)
        axs[i,j].set_ylabel("$\lambda_{}(t)$".format(k + 1))
        axs[i,j].set_title("Firing Rates")
        axs[i,j].set_xlabel("Time")
        k = k + 1
plt.tight_layout()
fig.savefig("TVpgGLM/fig/syn_tv_N10_raster.pdf")

#####################################
### Cross-correlation analysis (p3)##
#####################################
f_true = xcorr.xcorr(fr_true, dtmax=20)
f_est = xcorr.xcorr(fr_est, dtmax=20)

plt.figure()
fig, axs = plt.subplots(N, N, sharey=True)

for i in range(N):
    for j in range(N):
        axs[i, j].plot(f_true[i, j, :],color=color[0])
        axs[i, j].plot(f_est[i, j, :],color=color[1])
        axs[i, j].set_xticklabels([])
        axs[i, j].set_yticklabels([])

fig.savefig("TVpgGLM/fig/syn_tv_N10_xcorr.pdf")


###################################
##True and estimated weights (p4)##
###################################
plt.figure()
fig, axs = plt.subplots(N, N)

for i in range(N):
    for j in range(N):
        sns.tsplot(data=w_true[i,:,j,0],ax=axs[i, j], color=color[0])
        sns.tsplot(data=w_est[N_smpls // 2:,i,:,j, 0],ax=axs[i,j],color=color[1])
        axs[i,j].legend(loc="upper center", ncol=2, prop={'size':15})
        axs[i,j].set_xticklabels([])
        axs[i,j].set_yticklabels([])

plt.tight_layout()
fig.savefig("TVpgGLM/fig/syn_tv_N10_weights.pdf")