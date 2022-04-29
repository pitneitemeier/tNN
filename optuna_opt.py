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
lattice_sites = 14

def psi_init_z(spins):
    return (spins.sum(2) == spins.shape[2]).type(torch.get_default_dtype())

def psi_init_z_forward(spins, lattice_sites):
    return (spins.sum(1) == spins.shape[1]).type(torch.get_default_dtype()).unsqueeze(1)*(2**lattice_sites)

def psi_init_x(spins):
    lattice_sites = spins.shape[2]
    return torch.full_like(spins, 2**(- lattice_sites / 2))

def psi_init_x_forward(spins, lattice_sites):
    return torch.full_like(spins[:, :1], 1)

def trial_model(trial):
    #num_conv_layers = trial.suggest_int('num_conv_layers', 1, 4)
    num_conv_features = trial.suggest_int('num_conv_features', 8, 32, step=8)
    #psi_num_hidden = trial.suggest_int('psi_num_hidden', 2, 6)
    psi_hidden = trial.suggest_int('psi_hidden', 16, 256, step=32)
    #tNN_num_hidden = trial.suggest_int('tNN_num_hidden', 2, 6)
    tNN_hidden = trial.suggest_int('tNN_hidden', 16, 256, step=32)
    mult_size = trial.suggest_int('mult_size', 128, 1500, step=128)

    model = models.ParametrizedFeedForward(lattice_sites, num_h_params=1, learning_rate=1e-3, psi_init=psi_init_x_forward,
        act_fun=nn.GELU, kernel_size=3, num_conv_layers=3, num_conv_features=num_conv_features,
        tNN_hidden=tNN_hidden, tNN_num_hidden=3, mult_size=mult_size, psi_hidden=psi_hidden, psi_num_hidden=3, step_size=3, gamma=0.1)
    return model

def objective(trial: optuna.trial.Trial) -> float:
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
    h_param_range = [(0.15, 1.4)]

    ### The samplers that are used for training and validation. here fully random samples are used in training and full sums in validation
    train_sampler = sampler.RandomSampler(lattice_sites, 1)
    val_sampler = sampler.ExactSampler(lattice_sites)

    ### define conditions that have to be satisfied
    schrodinger = cond.schrodinger_eq_per_config(h_list=h_list, lattice_sites=lattice_sites, name='TFI}', 
        h_param_range=h_param_range, sampler=train_sampler, t_range=(0,3), epoch_len=int(3e6), exp_decay=False)
    val_cond = cond.Simple_ED_Validation(magn_op, lattice_sites, ED_magn, val_alpha, val_h_params, val_sampler, name_app=str(trial.number))

    env = tNN.Environment(train_condition=schrodinger, val_condition=val_cond, test_condition=val_cond,
        batch_size=200, val_batch_size=5, test_batch_size=5, num_workers=18)
    model = trial_model(trial)

    trainer = pl.Trainer(fast_dev_run=False, max_epochs=4,
                        accelerator="gpu", devices=1,
                        callbacks=[PyTorchLightningPruningCallback(trial, monitor="train_loss")])
    trainer.fit(model, env)

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
    study.optimize(objective, n_trials=15)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))