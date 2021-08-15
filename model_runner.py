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

def psi_init_x_forward(spins, lattice_sites):
    return torch.full_like(spins[:, :1], 2**(- lattice_sites / 2))

if __name__=='__main__':
    ### setting up hamiltonian
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

    ### define conditions that have to be satisfied
    schrodinger = cond.schrodinger_eq_per_config(h_list=h_list, lattice_sites=lattice_sites, name='TFI')
    val_cond = cond.ED_Validation(obs, lattice_sites, ED_magn, val_t_arr, val_h_params)
    
    from pytorch_lightning.callbacks import LearningRateMonitor
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='chkpts/', filename='TFI_8-{epoch:02d}-{val_loss:.2f}')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    val_sampler = sampler.ExactSampler(lattice_sites)
    train_sampler = sampler.RandomSampler(lattice_sites, 64)

    env = tNN.Environment(condition_list=[schrodinger], h_param_range=h_param_range, t_range=(0,3),
        train_sampler=train_sampler, val_sampler=val_sampler,
        batch_size=50, epoch_len=1e6, num_workers=48,
        val_condition=val_cond, val_h_params=val_h_params, val_t_arr=val_t_arr)
    model = models.init_fixed(lattice_sites=lattice_sites, num_h_params=1, learning_rate=1e-3, psi_init=psi_init_x_forward, patience=0)


    trainer = pl.Trainer(fast_dev_run=False, gpus=-1, max_epochs=50,
        auto_select_gpus=True, gradient_clip_val=.5,
        callbacks=[lr_monitor, checkpoint_callback],
        deterministic=False, progress_bar_refresh_rate=20,
        accelerator='ddp' ,plugins=DDPPlugin(find_unused_parameters=False) )
    '''
    trainer = pl.Trainer(resume_from_checkpoint='TFI8x_init_fixed_slurm1.ckpt', gpus=-1, max_epochs=50,
        auto_select_gpus=True, gradient_clip_val=.5,
        callbacks=[lr_monitor, checkpoint_callback],
        deterministic=False, progress_bar_refresh_rate=20,
        accelerator='ddp' ,plugins=DDPPlugin(find_unused_parameters=False) )
    '''
    trainer.fit(model=model, datamodule=env)
    trainer.save_checkpoint('TFI10x_init_fixed_slurm1.ckpt')
