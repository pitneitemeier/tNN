import pytorch_lightning as pl
import torch
from torch import nn
import activations as act
import Operator as op
import numpy as np
import matplotlib.pyplot as plt
import condition as cond
import tNN

@tNN.wave_function
class Model(pl.LightningModule):
    def __init__(self, lattice_sites, num_h_params, learning_rate):
        super().__init__()
        lattice_hidden = 64
        mult_size = 128
        self.lattice_sites = lattice_sites
        self.lr = learning_rate
        self.lattice_net = nn.Sequential(
        nn.Linear( lattice_sites, lattice_hidden),
        nn.CELU(),
        nn.Linear( lattice_hidden, mult_size),
        )

        tNN_hidden = 64
        self.tNN = nn.Sequential(
        nn.Linear(1 + num_h_params, tNN_hidden),
        nn.CELU(),
        nn.Linear(tNN_hidden, tNN_hidden),
        nn.CELU(),
        nn.Linear(tNN_hidden, mult_size),
        nn.CELU()
        )

        psi_hidden = int( mult_size / 2 )
        psi_type = torch.complex128 
        self.psi = nn.Sequential(
        act.Euler_act(),
        nn.Linear( psi_hidden, psi_hidden, dtype=psi_type),
        act.complex_celu(),
        nn.Linear( psi_hidden, psi_hidden, dtype=psi_type),
        act.complex_celu(),
        nn.Linear( psi_hidden, psi_hidden, dtype=psi_type),
        act.complex_celu(),
        nn.Linear( psi_hidden, 1, dtype=psi_type),
        )

    def forward(self, spins, alpha):
        lat_out = self.lattice_net(spins)
        t_out = self.tNN(alpha)
        psi_out = self.psi( (t_out * lat_out) )
        return psi_out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__=='__main__':
    ### setting up hamiltonian
    lattice_sites = 4
    h_param_range = [(0.45, 0.55)]

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
    #ext_params = np.array([0.9, 1., 1.1]).reshape(3,1)
    #ED_data = np.stack((ED_data_09, ED_data_10, ED_data_11))
    ED_data = np.expand_dims(ED_data_05, 0)
    val_h_params = np.array([.5]).reshape(1,1)

    ### define conditions that have to be satisfied
    schrodinger = cond.schrodinger_eq(h_list=h_list, lattice_sites=lattice_sites, name='TFI')
    init_cond = cond.init_observable(obs, lattice_sites=lattice_sites, name='sx init', weight=50)
    norm = cond.Norm(weight=2)
    val_cond = cond.ED_Validation(obs, lattice_sites, ED_data[:, :, 1], '', 'Mean_X_Magnetization')

    model = Model(lattice_sites=lattice_sites, num_h_params=1, learning_rate=1e-3)
    env = tNN.Environment(condition_list=[schrodinger, norm, init_cond], h_param_range=h_param_range, batch_size=200, 
        val_condition_list=[val_cond], val_h_params=val_h_params, val_t_arr=ED_data[:, :, 0])
    trainer = pl.Trainer(fast_dev_run=False, gpus=1, max_epochs=10)
    trainer.fit(model=model, datamodule=env)
    trainer.save_checkpoint('test.ckpt')
    model_reloaded = Model.load_from_checkpoint('test.ckpt')
    print('success!')