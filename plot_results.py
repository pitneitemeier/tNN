import utils
import models
import sampler
import Operator as op
import example_operators as ex_op
import numpy as np
import torch

def psi_init_x_forward(spins, lattice_sites):
    return torch.full_like(spins[:, :1], 2**(- lattice_sites / 2))

lattice_sites = 10

h1 = []
for l in range(lattice_sites):
    h1 += op.Sz(l) * (op.Sz((l+1) % lattice_sites))

h2 = []
for l in range(lattice_sites):
    h2 += op.Sx(l)

h_list = [h1, h2]

obs = ex_op.avg_magnetization(op.Sx, lattice_sites)
corr_list = [ex_op.avg_correlation(op.Sz, d+1, lattice_sites) for d in range(int(lattice_sites/2))]

### loading ED Data for Validation
folder = f'ED_data/TFI{lattice_sites}x/'
append = '_0.2_1.3.csv'
val_h_params = np.loadtxt(folder + 'h_params' + append, delimiter=',')
val_t_arr = np.loadtxt(folder + 't_arr' + append, delimiter=',')
ED_magn = np.loadtxt(folder + 'ED_magn' + append, delimiter=',')
ED_susc = np.loadtxt(folder + 'ED_susc' + append, delimiter=',')
ED_corr = np.loadtxt(folder + 'ED_corr' + append, delimiter=',').reshape(ED_magn.shape[0], int(lattice_sites/2), ED_magn.shape[1])
h_param_range = [(0.15, 1.4)]

model = models.ParametrizedFeedForward.load_from_checkpoint('TFI4x_FF_1.ckpt')
model.to('cuda')
print(model.device)
sampler = sampler.ExactSampler(lattice_sites)
utils.plot_results('TFIx', model, sampler, obs, corr_list, val_t_arr, val_h_params, 
    ED_magn, ED_susc, ED_corr, 'results/TFI4x/')



