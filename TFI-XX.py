import pytorch_lightning as pl
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.plugins import DDPPlugin
import torch
from torch import nn
from torch.utils import data
import activations as act
import Operator as op
import numpy as np
import matplotlib.pyplot as plt
import condition as cond
import tNN
import utils
import example_operators as ex_op
import models
import sampler


def psi_init_z(spins):
    return (spins.sum(2) == spins.shape[2]).type(torch.get_default_dtype())

def psi_init_z_forward(spins, lattice_sites):
    return (spins.sum(1) == spins.shape[1]).type(torch.get_default_dtype()).unsqueeze(1)*(2**lattice_sites)

def psi_init_x(spins):
    lattice_sites = spins.shape[2]
    return torch.full_like(spins, 2**(- lattice_sites / 2))

def psi_init_x_forward(spins, lattice_sites):
    return torch.full_like(spins[:, :1], 1)



if __name__=='__main__':
    ### setting up hamiltonian
    lattice_sites = 4
    init_polarization = 'x'
    pbc = False

    l_max = lattice_sites if pbc else lattice_sites - 1
    hZZ = []
    for l in range(l_max):
        hZZ += op.Sz(l) * (op.Sz((l + 1) % lattice_sites))
    hXX = []
    for l in range(l_max):
        hXX += op.Sx(l) * (op.Sx((l + 1) % lattice_sites))

    hZ = []
    for l in range(lattice_sites):
        hZ += op.Sz(l)
    hX = []
    for l in range(lattice_sites):
        hX += op.Sx(l)

    # ZZ + k * XX + g * X + h * Z @ (g=1.0, h=0.0, k varying)
    h_list = [hZZ + hX, hXX]

    magn_op = ex_op.avg_magnetization(op.Sx, lattice_sites)
    corr_list = [ex_op.avg_correlation(op.Sz, d+1, lattice_sites) for d in range(int(lattice_sites/2))]

    ### loading ED Data for Validation
    folder = f'ED_data/TFI_XX_{lattice_sites}/'
    append = '.csv'
    val_h_params = np.loadtxt(folder + 'h_params' + append, delimiter=',')
    # print(val_h_params.shape)
    val_t_arr = np.loadtxt(folder + 't_arr' + append, delimiter=',')
    # print(val_t_arr.shape)
    _t = np.broadcast_to(val_t_arr.reshape(1,-1,1), (val_h_params.shape[0], val_t_arr.shape[0], 1))
    # print(_t.shape)
    _h = np.broadcast_to(val_h_params.reshape(-1,1,1), (val_h_params.shape[0], val_t_arr.shape[0], 1))
    # print(_h.shape)
    val_alpha = np.concatenate((_t, _h), 2)

    print('validating on h= ', val_h_params)

    ED_magn = np.loadtxt(folder + 'ED_magn' + append, delimiter=',')
    # ED_susc = np.loadtxt(folder + 'ED_susc' + append, delimiter=',')
    # ED_corr = np.loadtxt(folder + 'ED_corr' + append, delimiter=',').reshape(ED_magn.shape[0], ED_magn.shape[1], int(lattice_sites/2))

    ### Defining the range for the external parameter that is trained
    h_param_range = [(0.05, 0.30)]

    ### The samplers that are used for training and validation. here fully random samples are used in training and full sums in validation
    train_sampler = sampler.RandomSampler(lattice_sites, 1)
    val_sampler = sampler.ExactSampler(lattice_sites)

    ### define conditions that have to be satisfied
    schrodinger = cond.schrodinger_eq_per_config(h_list=h_list, lattice_sites=lattice_sites, name='TFI_XZ', 
        h_param_range=h_param_range, sampler=train_sampler, t_range=(0,3), epoch_len=int(1e5), exp_decay=False)
    val_cond = cond.Simple_ED_Validation(magn_op, lattice_sites, ED_magn, val_alpha, val_h_params, val_sampler)

    env = tNN.Environment(train_condition=schrodinger, val_condition=val_cond, test_condition=val_cond,
        batch_size=100, val_batch_size=50, test_batch_size=2, num_workers=16)
    model = models.ParametrizedFeedForward(lattice_sites, num_h_params=1, learning_rate=1e-3, psi_init=psi_init_x_forward,
        act_fun=nn.GELU, kernel_size=4, num_conv_layers=3, num_conv_features=16,
        tNN_hidden=32, tNN_num_hidden=3, mult_size=512, psi_hidden=32, psi_num_hidden=3, step_size=2, gamma=0.1, init_decay=1)
    
    trainer = pl.Trainer(fast_dev_run=False, gpus=1, max_epochs=10, 
        auto_select_gpus=True, accelerator='ddp', plugins=DDPPlugin(find_unused_parameters=False))
    trainer.fit(model, env)

