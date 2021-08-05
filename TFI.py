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
torch.set_default_dtype(torch.float32)

def psi_init(spins):
    return (spins.sum(2) == spins.shape[2]).type(torch.get_default_dtype())

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
    folder = 'ED_data/TFI4x/'
    append = '_0.2_1.3.csv'
    val_h_params = np.loadtxt(folder + 'h_params' + append, delimiter=',')
    val_t_arr = np.loadtxt(folder + 't_arr' + append, delimiter=',')
    print(val_h_params)
    ED_magn = np.loadtxt(folder + 'ED_magn' + append, delimiter=',')
    ED_susc = np.loadtxt(folder + 'ED_susc' + append, delimiter=',')
    ED_corr = np.loadtxt(folder + 'ED_corr' + append, delimiter=',').reshape(ED_magn.shape[0], int(lattice_sites/2), ED_magn.shape[1])
    h_param_range = [(0.15, 1.35)]

    ### define conditions that have to be satisfied
    schrodinger = cond.schrodinger_eq(h_list=h_list, lattice_sites=lattice_sites, name='TFI')
    init_cond = cond.init_observable(obs, lattice_sites=lattice_sites, name='sx init', weight=75)
    #init_cond = cond.init_scalar_prod(psi_init, lattice_sites, 'z up', weight=1000)
    norm = cond.Norm(weight=25, norm_target=1)
    val_cond = cond.ED_Validation(obs, lattice_sites, ED_magn, val_t_arr, val_h_params)
    
    ### universal seed for deterministic behaviour
    #pl.seed_everything(42, workers=True)
    
    env = tNN.Environment(condition_list=[schrodinger, norm, init_cond], h_param_range=h_param_range, batch_size=50, epoch_len=2e3, 
        val_condition=val_cond, val_h_params=val_h_params, val_t_arr=val_t_arr, t_range=(0,3), num_workers=24)
    model = models.multConvDeep(lattice_sites=lattice_sites, num_h_params=1, learning_rate=1e-3)
    #model = models.multConvModel.load_from_checkpoint('tmp.ckpt')

    from pytorch_lightning.callbacks import LearningRateMonitor
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='chkpts/', filename='TFI_4-{epoch:02d}-{val_loss:.2f}')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(fast_dev_run=False, gpus=1, max_epochs=1,
        auto_select_gpus=True, gradient_clip_val=.5,
        callbacks=[lr_monitor, checkpoint_callback],
        deterministic=False, progress_bar_refresh_rate=5,
        accelerator='ddp')#, plugins=DDPPlugin(find_unused_parameters=False))
    trainer.fit(model=model, datamodule=env)
    #trainer.save_checkpoint('tmp.ckpt')


    #trainer = pl.Trainer(resume_from_checkpoint='tmp.ckpt', gpus=1, auto_select_gpus=True)
    #trainer.fit(model, env)


    #utils.plot_results('TFI Model', model, obs, corr_list, val_t_arr, val_h_params, ED_magn, ED_susc, ED_corr, 'results/TFI6x/')
    #print(ED_magn)