# Experimental results on retina datasets finding position
# load expermental data
import numpy as np
import pickle
import matplotlib.pyplot as plt
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
    lps, W_smpls, A_smpls, b_smpls, L_smpls= pickle.load(f)


#####################################
##Cross-correlation across distance##
#####################################
xc = np.corrcoef(Y.transpose())
plt.figure()
sns.heatmap(xc)
plt.figure()
sns.heatmap(W_mean[:,:,0])




