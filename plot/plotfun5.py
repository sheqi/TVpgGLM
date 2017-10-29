# experimental results of tv_model from selected two neurons
import sys
sys.path.append("/Users/roger/Dropbox/TVpgGLM-v1/TVpgGLM/libs")

import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from hips.plotting.colormaps import harvard_colors

sns.set_style("white")
paper_rc = {'lines.linewidth': 1, 'lines.markersize': 10,
            'font.size': 15, 'axes.labelsize':15, 'axes.titlesize':15,
            'xtick.labelsize': 15, 'ytick.labelsize': 15}
sns.set_context("paper", rc = paper_rc)
plt.ion()

color = harvard_colors()[0:10]

with open('TVpgGLM/results/exp_tv_N2.pickle', 'rb') as f:
     lps1, lps2, lps3,\
     W_mean1, W_mean2, W_mean3, W_std1, W_std2, W_std3, W_smpls, \
     Y_1st, Y_2nd, Y_12,\
     fr_mean1, fr_mean2, fr_mean3, fr_std1, fr_std2, fr_std3 = pickle.load(f)

########################
##static model weights##
########################
fig, ax = plt.subplots(1,2)
sns.heatmap(W_mean1[:,:,0], ax=ax[0], annot=True)
sns.heatmap(W_mean2[:,:,0], ax=ax[1], annot=True)
ax[0].set_yticks([])
ax[0].set_xticks([])
ax[1].set_yticks([])
ax[1].set_xticks([])
ax[0].set_xlabel('Post')
ax[0].set_ylabel('Pre')
ax[0].set_title('Before Learning')
ax[1].set_xlabel('Post')
ax[1].set_ylabel('Pre')
ax[1].set_title('During Learning')
plt.tight_layout()

plt.savefig("TVpgGLM/fig/exp_static_N2_weights.pdf")
####################
##tv_model weights##
####################
N_smpls = 100
fig, ax = plt.subplots(2,2)
for i in range(2):
    for j in range(2):
        sns.tsplot(data=W_smpls[N_smpls // 2:, i, :, j, 0], ax=ax[i, j], color=color[1])
        if i == 1:
            ax[i,j].set_xlabel('Time', fontweight="bold")
        if j == 0:
            ax[i,j].set_ylabel('Weights', fontweight="bold")

plt.tight_layout()
plt.savefig("TVpgGLM/fig/exp_tv_N2_weights.pdf")
#################################
##plot likelihood via iteration##
#################################
fig, ax = plt.subplots(1,3)
ax[0].plot(lps1)
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Log Likelihood")
ax[0].set_title('Before')
ax[1].plot(lps2)
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Log Likelihood")
ax[1].set_title('During')
ax[2].plot(lps3)
ax[2].set_xlabel("Iteration")
ax[2].set_ylabel("Log Likelihood")
ax[2].set_title('Before + During')
plt.tight_layout()
plt.savefig("TVpgGLM/fig/exp_N2_likhd.pdf")


