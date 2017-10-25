# plot simulation results of tv_model
import sys
sys.path.append("/Users/roger/Dropbox/TVpgGLM-v1/TVpgGLM/utils")

import pickle
import networkx as nx
import matplotlib.pyplot as plt
import xcorr
from plot_networks import draw_curvy_network
import seaborn as sns

sns.set_style("white")
paper_rc = {'lines.linewidth': 1, 'lines.markersize': 10,
            'font.size': 15, 'axes.labelsize':15,
            'xtick.labelsize': 15, 'ytick.labelsize': 15}
sns.set_context("paper", rc = paper_rc)
plt.ion()

# illustration (p1)
from hips.plotting.colormaps import harvard_colors
color = harvard_colors()[0:10]

G = nx.MultiDiGraph([(1, 1), (1, 2), (2, 1), (2, 2)])
pos = nx.spring_layout(G)
ax = plt.gca()
edge_width = [5, -5, -5, 5]
edge_color  = [color[0], color[1], color[1], color[0]]
draw_curvy_network(G, pos, ax, node_color='k',
                   node_edge_color='k', edge_width=edge_width,
                   edge_color=edge_color)
ax.autoscale()
plt.axis('equal')
plt.axis('off')
plt.show()

# raster plot-true (p2)


# raster plot-estimate (p3)


# Cross-correlation analysis (p4)
with open('TVpgGLM/results/sythetic_tv_N10.pickle', 'rb') as f:
    fr_true, w_true, fr_est, w_est = pickle.load(f)

N = 10
N_smpls = 50

f_true = xcorr.xcorr(fr_true, dtmax=20)
f_est = xcorr.xcorr(fr_est, dtmax=20)

fig, axs = plt.subplots(N, N, sharey=True)

for i in range(N):
    for j in range(N):
        axs[i, j].plot(f_true[i, j, :])
        axs[i, j].plot(f_est[i, j, :])
        axs[i, j].set_xticklabels([])
        axs[i, j].set_yticklabels([])

# True and estimated weights (p5)
fig, axs = plt.subplots(N, N)

for i in range(N):
    for j in range(N):
        sns.tsplot(data=w_true[i,:,j,0],ax=axs[i, j], color=color[0])
        sns.tsplot(data=w_est[N_smpls // 2:,i,:,j, 0],ax=axs[i,j],color=color[1])
        axs[i,j].legend(loc="upper center", ncol=2, prop={'size':15})
        axs[i,j].set_xticklabels([])
        axs[i,j].set_yticklabels([])