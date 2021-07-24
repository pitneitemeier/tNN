from typing import Sequence
from numpy.core.numeric import ones_like
import pytorch_lightning as pl
import torch
import Operator as op
import numpy as np
import Datasets
from torch.utils.data import DataLoader
from torch import nn
import activations as act
import tNN
import utils
from argparse import ArgumentParser


@tNN.wave_function
class Model(pl.LightningModule):
  def __init__(self, lattice_sites, ext_param_range):
    super().__init__()
    lattice_hidden = 64
    mult_size = 128
    self.lattice_net = nn.Sequential(
      nn.Linear( lattice_sites, lattice_hidden),
      nn.CELU(),
      nn.Linear( lattice_hidden, mult_size),
    )

    tNN_hidden = 64
    self.tNN = nn.Sequential(
      nn.Linear(1 + len(ext_param_range), tNN_hidden),
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
 

def psi_init(spins):
  return (spins.sum(-1) == spins.shape[-1]).unsqueeze(-1).type(torch.float64)


def main(args):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print('Using {} device'.format(device))
  torch.set_default_dtype(torch.float64)

  ### setting up hamiltonian
  lattice_sites = 4
  h2_range = [(0.45, 0.55)]

  h1 = []
  for l in range(lattice_sites):
    h1 = op.Sz(l) * (op.Sz((l+1) % lattice_sites)) + h1

  h2 = []
  for l in range(lattice_sites):
    h2 = op.Sx(l) + h2

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
  ext_params = np.array([.5]).reshape(1,1)

  ### Setting up Dataloaders
  val_data = Datasets.Val_Data(ED_data, ext_params)
  val_dataloader = DataLoader(val_data, batch_size=1, num_workers=24)
  val_iter = iter(val_dataloader)

  train_data = Datasets.Train_Data([h1, h2], h2_range)
  train_dataloader = DataLoader(dataset=train_data, batch_size=200, num_workers=24)
  data_iter = iter(train_dataloader)

  model = Model(lattice_sites, h2_range)
  env = tNN.Environment(model, lattice_sites, [h1, h2], obs, t_min=0, t_max=2, num_epochs=4, lr=1e-3, device=device, psi_init=psi_init)
  #print(model)

  trainer = pl.Trainer(fast_dev_run=False, gpus=1, max_epochs=1)
  trainer.fit(env, train_dataloader, val_dataloader)


if (__name__ == '__main__'):
  parser = ArgumentParser()
  #parser = pl.Trainer.add_argparse_args(parser)
  #parser.add_argument('--lattice_sites', type=int)
  args = parser.parse_args()
  main(args)
  #print('lattice_sites: ', args.lattice_sites)
  