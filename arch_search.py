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
#torch.set_default_dtype(torch.float32)

def psi_init_z(spins):
    return (spins.sum(2) == spins.shape[2]).type(torch.get_default_dtype())

def psi_init_z_forward(spins, lattice_sites):
    return (spins.sum(1) == spins.shape[1]).type(torch.get_default_dtype()).unsqueeze(1)

def psi_init_x(spins):
    lattice_sites = spins.shape[2]
    return torch.full_like(spins, 2**(- lattice_sites / 2))

def psi_init_x_forward(spins, lattice_sites):
    return torch.full_like(spins[:, :1], 2**(- lattice_sites / 2))



if __name__=='__main__':
    ### setting up hamiltonian
    lattice_sites = 4
    
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
    print('validating on h= ', val_h_params)
    ED_magn = np.loadtxt(folder + 'ED_magn' + append, delimiter=',')
    ED_susc = np.loadtxt(folder + 'ED_susc' + append, delimiter=',')
    ED_corr = np.loadtxt(folder + 'ED_corr' + append, delimiter=',').reshape(ED_magn.shape[0], int(lattice_sites/2), ED_magn.shape[1])
    h_param_range = [(0.15, 1.4)]

    ### define conditions that have to be satisfied
    schrodinger = cond.schrodinger_eq_per_config(h_list=h_list, lattice_sites=lattice_sites, name='TFI')
    val_cond = cond.ED_Validation(obs, lattice_sites, ED_magn, val_t_arr, val_h_params, MC_sampling=False)
    
    #train_sampler = sampler.ExactSampler(lattice_sites)
    train_sampler = sampler.RandomSampler(lattice_sites, 16)
    #val_sampler = sampler.MCMCSampler(lattice_sites, num_samples=256, steps_to_equilibrium=500)
    val_sampler = sampler.ExactSampler(lattice_sites)

    env = tNN.Environment(condition_list=[schrodinger], h_param_range=h_param_range, batch_size=50, epoch_len=2e5, 
        val_condition=val_cond, val_h_params=val_h_params, val_t_arr=val_t_arr, 
        train_sampler=train_sampler, val_sampler=val_sampler, t_range=(0,3), num_workers=48)
    
    test_name = 'ParametrizedSelfAttention'
    model = models.ParametrizedSelfAttention(lattice_sites, num_h_params=1, learning_rate=1e-3, psi_init=psi_init_x_forward,
        act_fun=nn.CELU, kernel_size=3, num_conv_layers=3, num_conv_features=32, 
        tNN_hidden=64, tNN_num_hidden=6, mult_size=512, psi_hidden=64, psi_num_hidden=3)

    from pytorch_lightning.callbacks import LearningRateMonitor
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='chkpts/', filename='TFI_4-{epoch:02d}-{val_loss:.2f}')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(fast_dev_run=False, gpus=[0,1], max_epochs=3,
        auto_select_gpus=True, gradient_clip_val=.5,
        callbacks=[lr_monitor, checkpoint_callback],
        deterministic=False, progress_bar_refresh_rate=5,
        accelerator='ddp', plugins=DDPPlugin(find_unused_parameters=False))
    trainer.fit(model, env)
    trainer.save_checkpoint(test_name+'.ckpt')