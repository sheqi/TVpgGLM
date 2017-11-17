# plot simulated 2-neuron network
from hips.plotting.colormaps import harvard_colors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

sns.set_style("white")
paper_rc = {'lines.linewidth': 2.5, 'lines.markersize': 10,
            'font.size': 15, 'axes.labelsize':15, 'axes.titlesize':15,
            'xtick.labelsize': 15, 'ytick.labelsize': 15}
sns.set_context("paper", rc = paper_rc)
plt.ion()

color = harvard_colors()[0:10]

with open('TVpgGLM/results/sythetic_tv_N2.pickle', 'rb') as f:
    w_true, w_est, Y, fr_est, fr_true, fr_std = pickle.load(f)


# Raster and firing rate plot
pltslice=slice(0, 500)
fig,axs = plt.subplots(1,2)
k = 0
for j in range(2):
    tn = np.where(Y[pltslice, k])[0]
    axs[j].plot(tn, np.ones_like(tn), 'ko', markersize=4)
    axs[j].plot(fr_est[pltslice,k],color=color[0])
    axs[j].plot(fr_true[pltslice, k],color=color[1])
        # sausage_plot(np.arange(pltslice.start, pltslice.stop),
        #             fr_est[pltslice,n],
        #             #fr_true[pltslice, n],
        #             3*fr_std[pltslice,n],
        #             sgax=axs[n],
        #             alpha=0.5)
    axs[j].set_ylim(-0.05, 1.1)
    axs[j].set_ylabel("$\lambda_{}(t)$".format(k + 1))
    axs[j].set_title("Firing Rates")
    axs[j].set_xlabel("Time")
    k = k + 1
plt.tight_layout()
fig.savefig("TVpgGLM/fig/syn_tv_N2_raster.pdf")

# Plot weights comparison
N = 2
N_samples = 50
fig, axs = plt.subplots(N, N)
for i in range(N):
    for j in range(N):
        sns.tsplot(data=w_true[i, 0:1000, j, 0],
                   ax=axs[i, j], color=color[4], alpha = 0.7)
        sns.tsplot(data=w_est[N_samples // 2:,i,0:1000,j, 0],
                   ax=axs[i,j],color=color[5], alpha = 0.7)
        if i == 1:
            axs[i,j].set_xlabel('Time', fontweight="bold")
        if j == 0:
            axs[i,j].set_ylabel('Weights',fontweight="bold")
axs[0,0].text(50, 1.0, "True", fontsize=13, fontweight="bold", color=color[4])
axs[0,0].text(50, 0.8, "Estimated", fontsize=13, fontweight="bold", color=color[5])

plt.tight_layout()
fig.savefig("TVpgGLM/fig/syn_tv_N2_weights.pdf")