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
from neptune.new.integrations.pytorch_lightning import NeptuneLogger
#torch.set_default_dtype(torch.float32)

def psi_init_z(spins):
    return (spins.sum(2) == spins.shape[2]).type(torch.get_default_dtype())

def psi_init_z_forward(spins, lattice_sites):
    return (spins.sum(1) == spins.shape[1]).type(torch.get_default_dtype()).unsqueeze(1)*(2**lattice_sites)

def psi_init_x(spins):
    lattice_sites = spins.shape[2]
    return torch.full_like(spins, 2**(- lattice_sites / 2))

def psi_init_x_forward(spins, lattice_sites):
    return torch.full_like(spins[:, :1], 1)

def ramp(t, t_ramp_end=1.5, h_max=1.5):
    return h_max * torch.clamp(t/t_ramp_end, 0, 1)

def pulse(t, loc=1.5, scale=.5, frec=20):
    return torch.exp(-.5*(t-loc)**2/(scale**2)) * torch.sin(t * frec)


if __name__=='__main__':
    ### setting up hamiltonian
    lattice_sites = 10
    init_polarization = 'ramp'
    
    h1 = []
    for l in range(lattice_sites):
        h1 += op.Sx(l) * (op.Sx((l+1) % lattice_sites))

    h2 = []
    for l in range(lattice_sites):
        h2 += op.Sz(l)
    
    h_list = [h1, h2]

    magn_op = ex_op.avg_magnetization(op.Sz, lattice_sites)
    corr_list = [ex_op.avg_correlation(op.Sx, d+1, lattice_sites) for d in range(int(lattice_sites/2))]

    ### loading ED Data for Validation
    folder = f'ED_data/TFI{lattice_sites}{init_polarization}/'
    val_alpha = np.expand_dims(np.loadtxt(folder + 'alpha.csv', delimiter=','), axis=0)
    val_names = [1.5]
    ED_magn = np.expand_dims(np.loadtxt(folder + 'ED_magn.csv', delimiter=','), axis=0)
    ED_corr = np.expand_dims(np.loadtxt(folder + 'ED_corr.csv', delimiter=','), axis=0)
    val_h_params = ['h(t)']

    tot_samples_epoch = 2e7
    samples_per_alpha = 1
    tot_batch_size = 4000
    batch_size = int(tot_batch_size/samples_per_alpha)
    epoch_len = int(tot_samples_epoch/samples_per_alpha)
    steps = int(epoch_len/batch_size)

    h_param_dict = {'tot_samples':tot_samples_epoch, 'tot_batch':tot_batch_size, 'samples_per_alpha':samples_per_alpha}
    train_sampler = sampler.RandomSampler(lattice_sites, samples_per_alpha)
    #val_sampler = sampler.MCMCSamplerChains(lattice_sites, num_samples=64, steps_to_equilibrium=100)
    #val_sampler = sampler.MCMCSampler(lattice_sites, num_samples=256, steps_to_equilibrium=50)
    val_sampler = sampler.ExactSampler(lattice_sites)

    ### define conditions that have to be satisfied
    schrodinger = cond.schrodinger_eq_time_dep(h_list=h_list, lattice_sites=lattice_sites, name='TFI', 
        h_func=ramp, sampler=train_sampler, t_range=(0,3), epoch_len=epoch_len)
    val_cond = cond.ED_Validation(magn_op, lattice_sites, ED_magn, val_alpha, val_names, val_sampler)
    test_cond = cond.Time_dep_Test(magn_op, corr_list, h_list, lattice_sites, ED_magn, ED_corr, val_alpha, val_h_params, val_sampler, 
        name=f'TFI {init_polarization}', plot_folder=f'results/TFI{lattice_sites}{init_polarization}/',
        plot_fmt='.pdf', magn_op_name='\sigma^x', corr_op_name='\sigma^z')
    #test_cond = cond.magn_surface(magn_op, [0.2,1.2], [0,3], lattice_sites, val_sampler, 'TFI')

    env = tNN.Environment(train_condition=schrodinger, val_condition=val_cond, test_condition=test_cond,
        batch_size=batch_size, val_batch_size=50, test_batch_size=1, num_workers=24)
    #model = models.ParametrizedFeedForward(lattice_sites, num_h_params=1, learning_rate=1e-3, psi_init=psi_init_x_forward,
    #    act_fun=nn.GELU, kernel_size=3, num_conv_layers=3, num_conv_features=24,
    #    tNN_hidden=128, tNN_num_hidden=3, mult_size=1024, psi_hidden=80, psi_num_hidden=3, step_size=1, gamma=0.1, init_decay=1)
    model = models.ParametrizedFeedForward.load_from_checkpoint('TFI10z_FF_2slurm.ckpt')
    model.lr=1e-3
    model.step_size = 1

    from pytorch_lightning.callbacks import LearningRateMonitor
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='chkpts/', filename=f'TFI_{lattice_sites}{init_polarization}_'+model.name+'-{epoch:02d}-{val_loss:.6f}')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    neptune_logger = NeptuneLogger(
            api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMjAxNDViMi1mOWFjLTRlODEtYmZiZi02MDA4ZDIyMTYxODEifQ==',
            project='pitneitemeier/tNN', 
            name=f'TFI{init_polarization}',
            close_after_fit=False
        )
    neptune_logger.log_hyperparams(h_param_dict)

    trainer = pl.Trainer(fast_dev_run=False, gpus=[0,1], max_epochs=3,
        auto_select_gpus=True, gradient_clip_val=.5,
        callbacks=[lr_monitor, checkpoint_callback],
        deterministic=False, logger=neptune_logger,
        accelerator='ddp', plugins=DDPPlugin(find_unused_parameters=False))
    #trainer.tune(model, env)
    trainer.fit(model, env)
    trainer.save_checkpoint(f'TFI{lattice_sites}{init_polarization}_FF_1.ckpt')
    trainer.test(model=model, datamodule=env)
