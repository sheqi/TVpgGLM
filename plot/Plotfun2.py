import sys;

sys.path.append("/Users/Roger/Dropbox/pyglm-master/pyglm/")
sys.path.append("/Users/Roger/Dropbox/pyglm-master/pyglm/utils/")

import matplotlib.pyplot as plt
import numpy as np
import pickle
from hips.plotting.colormaps import harvard_colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

sns.set_style("white")
paper_rc = {'lines.linewidth': 2.5, 'lines.markersize': 10, 'font.size': 15,
            'axes.labelsize':15, 'xtick.labelsize': 15, 'ytick.labelsize': 15}
sns.set_context("paper", rc = paper_rc)
plt.ion()

# plot: sythetic_true_location
with open('TVpgGLM/results/sythetic_true_location2.pickle', 'rb') as f:
    lps, W_true, W_smpls, A_smpls, L_true, L_smples = pickle.load(f)

N = 9
N_samples = 2000
k = 0
fig = plt.figure()
handles = []

## True weighted adjacency matrix
ax0 = fig.add_subplot(131, aspect="equal")
im0 = ax0.imshow(W_true[:,:,0],vmin=-3, vmax=3, cmap="RdBu_r", interpolation="nearest")
ax0.set_yticks([])
ax0.set_xticks([])
ax0.set_xlabel('Post')
ax0.set_ylabel('Pre')
ax0.set_title(r'True $A \odot W$')

# Colorbar
divider = make_axes_locatable(ax0)
cbax = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(cbax)
plt.colorbar(im0, cax=cbax)
handles.append(im0)

## Est weighted adjacency matrix
ax1 = fig.add_subplot(132, aspect="equal")
W_mean = W_smpls[N_samples // 2:].mean(0)
im1 = ax1.imshow(W_mean[:,:,0],vmin=-3, vmax=3, cmap="RdBu_r", interpolation="nearest")
ax1.set_yticks([])
ax1.set_xticks([])
ax1.set_xlabel('Post')
ax1.set_ylabel('Pre')
ax1.set_title(r'MCMC $\mathbb{E}[A \odot W]$')

# Colorbar
divider = make_axes_locatable(ax1)
cbax = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(cbax)
plt.colorbar(im1, cax=cbax)
handles.append(im1)

## Latent location
color = harvard_colors()[0:10]
ax2 = fig.add_subplot(133, aspect="equal")

for i in range(N):
    ax2.scatter(L_smples[-500:, i, 0],
               L_smples[-500:, i, 1], alpha=0.1, s=50, c=color[k])
    k += 1

ax2.scatter(L_true[:, 0], L_true[:, 1], s=150, c=color, edgecolor='0',
           lw=1, marker=(5, 1))

b = np.amax(abs(L_true)) + L_true[:].std() / 2.0

# Plot grids for origin
ax2.plot([1.24, 1.24], [-b, b+0.5], ':k', lw=0.2)
ax2.plot([-b, b+0.5], [1.24, 1.24], ':k', lw=0.2)

# Set the limits
ax2.set_xlim([-1, b+0.5])
ax2.set_ylim([-1, b+0.5])

# Labels
ax2.set_xlabel(r'$\ell_{1}$[a.u.]',labelpad=0.2)
ax2.set_ylabel(r'$\ell_{2}$[a.u.]',labelpad=0.2)
ax2.set_title('True & Inferred Locations')

fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.5, hspace=None)

