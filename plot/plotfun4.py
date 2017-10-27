# Experimental results on retina datasets finding position
# load expermental data
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from hips.plotting.colormaps import harvard_colors

sns.set_style("white")
paper_rc = {'lines.linewidth': 1, 'lines.markersize': 5,
            'font.size': 15, 'axes.labelsize':15, 'axes.titlesize':15,
            'xtick.labelsize': 15, 'ytick.labelsize': 15}
sns.set_context("paper", rc = paper_rc)
plt.ion()

color = harvard_colors()[0:10]

with open('TVpgGLM/results/exp_static_electrode.pickle', 'rb') as f:
    lps, W_smpls, A_smpls, b_smpls, L_smpls, L_true_norm, Y, D_mat = pickle.load(f)

N = 28
#####################
##xcorr vs distance##
#####################
xc = np.corrcoef(Y.transpose())
plt.scatter(D_mat.reshape(-1,1),xc.reshape(-1,1))
plt.xlabel('Distance(mm)')
plt.ylabel('Correlation Coefficient')
plt.tight_layout()


##############################
##True vs Estimated Position##
##############################
k = 0
for i in range(N):
    plt.scatter(L_smpls[-50:, i, 0], L_smpls[-50:, i, 1], s=100,
               alpha=0.5,c=np.random.rand(3,))
    k += 1

plt.scatter(L_true_norm[:,0],L_true_norm[:,1],marker='s', c='k')




