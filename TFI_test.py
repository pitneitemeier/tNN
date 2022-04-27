import pytorch_lightning as pl
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.plugins import DDPPlugin
import torch
from torch import nn
import Operator as op
import numpy as np
import condition as cond
import tNN
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

name = "_check_1"

if __name__=='__main__':
    ### setting up hamiltonian
    lattice_sites = 10
    init_polarization = 'x'
    
    h1 = []
    for l in range(lattice_sites):
        h1 += op.Sz(l) * (op.Sz((l+1) % lattice_sites))

    h2 = []
    for l in range(lattice_sites):
        h2 += op.Sx(l)
    
    h_list = [h1, h2]

    magn_op = ex_op.avg_magnetization(op.Sx, lattice_sites)
    corr_list = [ex_op.avg_correlation(op.Sz, d+1, lattice_sites) for d in range(int(lattice_sites/2))]

    ### loading ED Data for Validation
    folder = f'ED_data/TFI{lattice_sites}{init_polarization}/'
    append = '_0.2_1.3.csv'
    val_h_params = np.loadtxt(folder + 'h_params' + append, delimiter=',')
    val_t_arr = np.loadtxt(folder + 't_arr' + append, delimiter=',')
    val_alpha = np.concatenate(
        (np.broadcast_to(val_t_arr.reshape(1,-1,1), (val_h_params.shape[0], val_t_arr.shape[0], 1)), 
        np.broadcast_to(val_h_params.reshape(-1,1,1), (val_h_params.shape[0], val_t_arr.shape[0], 1))), 2)

    print('validating on h= ', val_h_params)

    ED_magn = np.loadtxt(folder + 'ED_magn' + append, delimiter=',')

    ### Defining the range for the external parameter that is trained
    h_param_range = [(0.15, 1.35)]

    ### The samplers that are used for training and validation. here fully random samples are used in training and full sums in validation
    train_sampler = sampler.RandomSampler(lattice_sites, 1)
    val_sampler = sampler.ExactSampler(lattice_sites)

    ### define conditions that have to be satisfied
    schrodinger = cond.schrodinger_eq_per_config(h_list=h_list, lattice_sites=lattice_sites, name='TFI}', 
        h_param_range=h_param_range, sampler=train_sampler, t_range=(0,3), epoch_len=4000, exp_decay=False)
    val_cond = cond.Simple_ED_Validation(magn_op, lattice_sites, ED_magn, val_alpha, val_h_params, val_sampler, name_app=name)

    env = tNN.Environment(train_condition=schrodinger, val_condition=val_cond, test_condition=val_cond,
        batch_size=4000, val_batch_size=5, test_batch_size=1, num_workers=36)
    #model = models.ParametrizedFeedForward(lattice_sites, num_h_params=1, learning_rate=1e-3, psi_init=psi_init_x_forward,
    #    act_fun=nn.GELU, kernel_size=3, num_conv_layers=3, num_conv_features=32,
    #    tNN_hidden=128, tNN_num_hidden=3, mult_size=1024, psi_hidden=64, psi_num_hidden=3, step_size=5, gamma=0.1)
    model = models.ParametrizedFeedForward.load_from_checkpoint("TFI10x_FF_1.ckpt")
    model.lr = 0

    trainer = pl.Trainer(accelerator="gpu", devices=2, strategy="ddp", max_epochs=1)
    trainer.fit(model, env)
    #trainer.test(model=model, datamodule=env)
