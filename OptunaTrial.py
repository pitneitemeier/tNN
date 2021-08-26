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
import os
import argparse
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
#torch.set_default_dtype(torch.float32)
LATTICE_SITES = 8

def psi_init_z(spins):
    return (spins.sum(2) == spins.shape[2]).type(torch.get_default_dtype())

def psi_init_x_forward(spins, lattice_sites):
    return torch.full_like(spins[:, :1], 2**(- lattice_sites / 2))

def trial_model(trial):
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 3)
    num_conv_features = trial.suggest_int('num_conv_features', 8, 32)
    psi_num_hidden = trial.suggest_int('psi_num_hidden', 2, 8)
    psi_hidden = trial.suggest_int('psi_hidden', 16, 128)
    tNN_num_hidden = trial.suggest_int('tNN_num_hidden', 2, 8)
    tNN_hidden = trial.suggest_int('tNN_hidden', 16, 128)
    mult_size = trial.suggest_int('mult_size', 128, 1024)

    model = models.parametrized(LATTICE_SITES, num_h_params=1, learning_rate=1e-3, psi_init=psi_init_x_forward,
        act_fun=nn.CELU, kernel_size=3, num_conv_layers=num_conv_layers, num_conv_features=num_conv_features, 
        tNN_hidden=tNN_hidden, tNN_num_hidden=tNN_num_hidden, mult_size=mult_size, psi_hidden=psi_hidden, psi_num_hidden=psi_num_hidden)
    
    return model

def objective(trial: optuna.trial.Trial) -> float:
    ### setting up hamiltonian

    
    h1 = []
    for l in range(LATTICE_SITES):
        h1 += op.Sz(l) * (op.Sz((l+1) % LATTICE_SITES))

    h2 = []
    for l in range(LATTICE_SITES):
        h2 += op.Sx(l)
    
    h_list = [h1, h2]

    obs = ex_op.avg_magnetization(op.Sx, LATTICE_SITES)
    corr_list = [ex_op.avg_correlation(op.Sz, d+1, LATTICE_SITES) for d in range(int(LATTICE_SITES/2))]

    ### loading ED Data for Validation
    folder = f'ED_data/TFI{LATTICE_SITES}x/'
    append = '_0.2_1.3.csv'
    val_h_params = np.loadtxt(folder + 'h_params' + append, delimiter=',')
    val_t_arr = np.loadtxt(folder + 't_arr' + append, delimiter=',')
    print('validating on h= ', val_h_params)
    ED_magn = np.loadtxt(folder + 'ED_magn' + append, delimiter=',')
    ED_susc = np.loadtxt(folder + 'ED_susc' + append, delimiter=',')
    ED_corr = np.loadtxt(folder + 'ED_corr' + append, delimiter=',').reshape(ED_magn.shape[0], int(LATTICE_SITES/2), ED_magn.shape[1])
    h_param_range = [(0.15, 1.4)]

    model = trial_model(trial)

    ### define conditions that have to be satisfied
    schrodinger = cond.schrodinger_eq_per_config(h_list=h_list, lattice_sites=LATTICE_SITES, name='TFI')
    val_cond = cond.ED_Validation(obs, LATTICE_SITES, ED_magn, val_t_arr, val_h_params)
    
    #samplers
    train_sampler = sampler.RandomSampler(LATTICE_SITES, 16)
    val_sampler = sampler.ExactSampler(LATTICE_SITES)

    #datamodule
    env = tNN.Environment(condition_list=[schrodinger], h_param_range=h_param_range, batch_size=50, epoch_len=2e5, 
        val_condition=val_cond, val_h_params=val_h_params, val_t_arr=val_t_arr, 
        train_sampler=train_sampler, val_sampler=val_sampler, t_range=(0,3), num_workers=48)

    trainer = pl.Trainer(fast_dev_run=False, gpus=1, max_epochs=2,
        auto_select_gpus=True, gradient_clip_val=.5,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="train_loss")],
        deterministic=False, progress_bar_refresh_rate=5,
        )#accelerator='ddp', plugins=DDPPlugin(find_unused_parameters=False))
    trainer.fit(model, datamodule=env)

    return trainer.callback_metrics["train_loss"].item()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=1)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


    