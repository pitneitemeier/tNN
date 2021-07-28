import pytorch_lightning as pl
import torch
from torch import nn
import activations as act
import Operator as op
import numpy as np
import matplotlib.pyplot as plt
import condition as cond
import tNN
import utils
import example_operators as ex_op
import models


def psi_init(spins):
    return (spins.sum(2) == spins.shape[2]).type(torch.float64)

if __name__=='__main__':
    ### setting up hamiltonian
    lattice_sites = 4
    

    h1 = []
    for l in range(lattice_sites):
        h1 = op.Sz(l) * (op.Sz((l+1) % lattice_sites)) + h1

    h2 = []
    for l in range(lattice_sites):
        h2 = op.Sx(l) + h2
    
    h_list = [h1, h2]

    obs = []
    for l in range(lattice_sites):
        obs = op.Sx(l) * (1 / lattice_sites) + obs
        
    corr_list = [ex_op.avg_correlation(op.Sz, d+1, lattice_sites) for d in range(int(lattice_sites/2))]

    ### Setting up datasets
    folder = 'ED_data/'
    file = 'ED_data4_'
    fmt = '.csv'
    path = folder + file
    ED_data_02 = np.loadtxt(path + '02' + fmt, delimiter=',')
    ED_data_05 = np.loadtxt(path + '05' + fmt, delimiter=',')
    ED_data_07 = np.loadtxt(path + '07' + fmt, delimiter=',')
    ED_data_09 = np.loadtxt(path + '09' + fmt, delimiter=',')
    ED_data_10 = np.loadtxt(path + '10' + fmt, delimiter=',')
    ED_data_11 = np.loadtxt(path + '11' + fmt, delimiter=',')
    ED_data_12 = np.loadtxt(path + '12' + fmt, delimiter=',')
    ED_data_13 = np.loadtxt(path + '13' + fmt, delimiter=',')

    #ED_data = np.stack((ED_data_02, ED_data_05, ED_data_07, ED_data_10, ED_data_13))
    h_param_range = [(0.8, 1.2)]
    val_h_params = np.array([0.9, 1., 1.1]).reshape(3,1)
    ED_data = np.stack((ED_data_09, ED_data_10, ED_data_11))
    #ED_data = np.expand_dims(ED_data_05, 0)
    #val_h_params = np.array([.5]).reshape(1,1)

    ### define conditions that have to be satisfied
    schrodinger = cond.schrodinger_eq(h_list=h_list, lattice_sites=lattice_sites, name='TFI')
    init_cond = cond.init_observable(obs, lattice_sites=lattice_sites, name='sx init', weight=50)
    #init_cond = cond.init_scalar_prod(psi_init, lattice_sites, 'z up', weight=50)
    norm = cond.Norm(weight=2, norm_target=1)
    val_cond = cond.ED_Validation(obs, lattice_sites, ED_data[:, :, 1], '', 'Mean_X_Magnetization')
 
    #model = models.multConvModel(lattice_sites=lattice_sites, num_h_params=1, learning_rate=1e-4)
    model = models.multConvModel.load_from_checkpoint('TFI_model4_2.ckpt')
    model.lr = 5e-6
    env = tNN.Environment(condition_list=[schrodinger, norm, init_cond], h_param_range=h_param_range, batch_size=200, 
        val_condition_list=[val_cond], val_h_params=val_h_params, val_t_arr=ED_data[:, :, 0], t_range=(0,1.5), num_workers=24)
    trainer = pl.Trainer(fast_dev_run=False, gpus=1, max_epochs=10, stochastic_weight_avg=True)
    trainer.fit(model=model, datamodule=env)
    trainer.save_checkpoint('TFI_model4_2')

'''
    t_arr = torch.from_numpy(ED_data[0, :, 0]).reshape(-1, 1, 1)
    h_param = torch.full_like(t_arr, 0.5)
    alpha = torch.cat((t_arr, h_param), dim=2)
    alpha = alpha.repeat(1, model.spins.shape[1], 1)
    alpha.requires_grad = True
    magn = model.measure_observable(alpha, obs, 4)
    susc = utils.get_susceptibility(magn, alpha)
    t_arr = t_arr[:,0,0]

    corr = [model.measure_observable(alpha, corr, lattice_sites).detach() for corr in corr_list]
    corr = torch.stack(corr)
    fig, ax = plt.subplots(3, 2, figsize=(10,6), sharex='col', gridspec_kw={'width_ratios': [20, 1]})
    ax[0,1].axis('off')
    ax[1,1].axis('off')
    ax[0, 0].plot(t_arr, magn.detach())
    ax[1, 0].plot(t_arr, susc.detach())
    y = np.arange(0,lattice_sites/2 + 1) + 0.5
    X, Y = np.meshgrid(t_arr, y)
    c = ax[2,0].pcolor(X, Y, corr[:, :-1])
    ax[2,0].set_yticks(y[:-1] + 0.5)
    plt.colorbar(c, cax=ax[2,1])
    ax[2,0].set_xlabel('ht')
    ax[0,0].set_ylabel(r'$ \langle S^x \rangle$', fontsize=13)
    ax[1,0].set_ylabel(r'$ \frac{\partial \langle S^x \rangle}{\partial h}$', fontsize=17)
    ax[2,0].set_ylabel(r'$\langle S^z_i \cdot S^z_{i+d} \rangle $'+ '\n d', fontsize=13)
    fig.tight_layout()
    fig.savefig('test.png')
'''