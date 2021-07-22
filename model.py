import pytorch_lightning as pl
import torch
from torch._C import dtype
from torch.cuda import init
from torch.nn.modules.linear import Linear
import Operator as op
import utils
import numpy as np
import Datasets
from torch.utils.data import Dataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt
import activations as act
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
g_dtype = torch.float64



###setting up hamiltonian ###
lattice_sites = 4

h2_range = [(0.45, 0.55)]

h1 = []
for l in range(lattice_sites):
  h1 = op.Sz(l) * (op.Sz((l+1) % lattice_sites)) + h1

h2 = []
for l in range(lattice_sites):
  h2 = op.Sx(l) + h2

o = []
for l in range(lattice_sites):
  o = op.Sx(l) * (1 / lattice_sites) + o

h1_mat = utils.get_total_mat_els(h1, lattice_sites)
h2_mat = utils.get_total_mat_els(h2, lattice_sites)
h_mat = torch.cat((h1_mat, h2_mat), dim=-2)
o_mat = utils.get_total_mat_els(o, lattice_sites)

h_map = utils.get_map(h1 + h2, lattice_sites)
o_map = utils.get_map(o, lattice_sites)


'''
One spin Operator
h = op.Sz(0) + []
o = op.Sx(0) + []

h_mat = utils.get_total_mat_els(h, lattice_sites)
o_mat = utils.get_total_mat_els(o, lattice_sites)

h_map = utils.get_map(h, lattice_sites)
o_map = utils.get_map(o, lattice_sites)
'''

###Setting up datasets

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

val_data = Datasets.Val_Data(lattice_sites, ED_data, ext_params, g_dtype)
val_dataloader = DataLoader(val_data, batch_size=1, num_workers=24)
val_iter = iter(val_dataloader)

train_data = Datasets.Train_Data(lattice_sites, [len(h1), len(h2)], h2_range, g_dtype)
train_dataloader = DataLoader(dataset=train_data, batch_size=200, num_workers=24)
data_iter = iter(train_dataloader)


class Model(pl.LightningModule):

  def __init__(self, lattice_sites, h_mat, h_map, o_mat, o_init_map, t_min, t_max):
    '''
    Initializer for neural net
    Parameters
    __________
    lattice_sites: int
    '''
    super().__init__()

    lattice_hidden = 64
    mult_size = 128
    self.lattice_net = nn.Sequential(
      nn.Linear( lattice_sites, lattice_hidden, dtype=g_dtype),
      nn.CELU(),
      nn.Linear( lattice_hidden, mult_size, dtype=g_dtype),
    )

    tNN_hidden = 64
    tNN_type = g_dtype
    self.tNN = nn.Sequential(
      nn.Linear(2, tNN_hidden, dtype=tNN_type),
      nn.CELU(),
      nn.Linear(tNN_hidden, tNN_hidden, dtype=tNN_type),
      nn.CELU(),
      nn.Linear(tNN_hidden, mult_size, dtype=tNN_type),
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

    self.flatten_spins = nn.Flatten(end_dim=-1)
    self.h_map = h_map.to(device)
    self.o_init_map = o_init_map.to(device)
    self.h_mat = h_mat.to(device)
    self.o_mat = o_mat.to(device)
    self.t_min = t_min
    self.t_max = t_max

  def forward(self, spins, alpha):
    #unsqueeze since circular padding needs tensor of dim 3
    lat_out = self.lattice_net(spins)
    
    t_out = self.tNN(alpha)

    psi_out = self.psi( (t_out * lat_out) )
    return psi_out
    
  def call_forward(self, spins, alpha):
    '''
    makes forward callable with (num_alpha_configs, num_spin_configs)
    Parameters
    __________
    spins: tensor, dtype=float
      tensor of input spins to wave function 
      shape = (num_spin_configs, num_alpha_configs, lattice_sites)
    alpha: tensor, dtype=float
      other inputs to hamiltonian e.g. (time, ext_param) 
      shape = (num_spin_configs, num_alpha_configs, num_inputs)

    Returns
    _______
    psi: tensor, dtype=complex
      wave function for a combination of (spins, alpha) 
      size = (num_spin_configs, num_alpha_configs, 1)
    '''
    spin_shape = spins.shape
    alpha_shape = alpha.shape
    
    spins = torch.flatten(spins, end_dim=-2)
    alpha = torch.flatten(alpha, end_dim=-2)
    
    psi = self(spins, alpha)
    return psi.reshape( spin_shape[0], spin_shape[1], 1)

  def call_forward_sp(self, sprimes, alpha):
    '''
    makes forward callable with (num_alpha_configs, num_spin_configs, num_sprimes)
    Parameters
    __________
    spins: tensor, dtype=float
      tensor of input spins to wave function 
      shape = (num_spin_configs, num_alpha_configs, num_sprimes, lattice_sites)
    alpha: tensor, dtype=float
      other inputs to hamiltonian e.g. (time, ext_param) are broadcasted to s' shape
      shape = (num_spin_configs, num_alpha_configs, num_inputs)

    Returns
    _______
    psi: tensor, dtype=complex
      wave function for a combination of (spins, alpha) 
      size = (num_spin_configs, num_alpha_configs, num_sprimes, 1)
    '''
    sprimes_shape = sprimes.shape
    alpha_shape = alpha.shape
    alpha = alpha.unsqueeze(2)
    alpha = alpha.broadcast_to(alpha_shape[0], alpha_shape[1], sprimes_shape[2], alpha_shape[2])

    sprimes = torch.flatten(sprimes, end_dim=-2)
    alpha = torch.flatten(alpha, end_dim=-2)
    
    psi = self(sprimes, alpha)
    return psi.reshape( sprimes_shape[0], sprimes_shape[1], sprimes_shape[2], 1)
  
  def training_step(self, batch, batch_idx):
    spins, alpha, alpha_0, ext_param_scale = batch

    #calculate dynamic end time and decay rate for loss
    n = int(self.current_epoch / 3) + 1
    N = 10
    t_max = self.t_min + (self.t_max - self.t_min) * np.log(10 * n / N + 1) / np.log( 11 )
    #t_max = self.t_max
    loss_weight = 1e-2 / (t_max/self.t_max + 1e-2)
    if (batch_idx == 0):
      print(loss_weight)
    #update uniformly sampled t_val to new t range
    alpha[:, :, 0] = (t_max - self.t_min) * alpha[:, :, 0] + self.t_min

    #gradient needed for dt_psi
    alpha.requires_grad = True
    self.log('end time', t_max, prog_bar=True)
    
    #get psi(s, alpha)
    psi_s = self.call_forward(spins, alpha)
    #get s' and psi(s', alpha) for h
    sp_h = utils.get_sp(spins, self.h_map)
    psi_sp_h = self.call_forward_sp(sp_h, alpha)
    #calc h_loc for h
    h_loc = utils.calc_Oloc(psi_sp_h, self.h_mat, spins, ext_param_scale)
    dt_psi_s = utils.calc_dt_psi(psi_s, alpha)
    
    #get s' and psi(s', alpha) for o at t=0
    sp_o = utils.get_sp(spins, self.o_init_map)
    psi_sp_o = self.call_forward_sp(sp_o, alpha_0)
    psi_s_0 = self.call_forward(spins, alpha_0)
    
    #calc o_loc for o
    o_loc = utils.calc_Oloc(psi_sp_o, self.o_mat, spins)

    #calc loss
    loss, schrodinger, init_cond, norm = utils.train_loss2(dt_psi_s, h_loc, psi_s, psi_s_0, o_loc, alpha, loss_weight)
    self.log('schrodinger', schrodinger, prog_bar=True, logger=True)
    self.log('init_cond', init_cond, prog_bar=True, logger=True)
    self.log('norm', norm, prog_bar=True, logger=True)
    self.log('train_loss', loss, logger=True)

    return {'loss': loss}
  
  def validation_step(self, batch, batch_idx):
    spins, alpha, o_target = batch
    n = int(self.current_epoch / 3) + 1
    N = 10
    t_max = self.t_min + (self.t_max - self.t_min) * np.log(10 * n / N + 1) / np.log( 11 ) + 0.2
    #t_max = self.t_max    
    max_ind = torch.argmin(torch.abs(alpha[0, :, 0, 0] - t_max))
    spins = spins[0, :max_ind, :, :]
    alpha = alpha[0, :max_ind, :, :]
    o_target = o_target[0, :max_ind]
    

    psi_s = self.call_forward(spins, alpha)
    sp_o = utils.get_sp(spins, self.o_init_map)
    psi_sp_o = self.call_forward_sp(sp_o, alpha)
    o_loc = utils.calc_Oloc(psi_sp_o, self.o_mat, spins)
    val_loss, observable = utils.val_loss(psi_s, o_loc, o_target)
    #self.log('val_loss', val_loss, prog_bar=True, logger=True)

    #print('obs_shape', observable.shape)
    #print('alpha_shape', alpha.shape)
    
    return {'observable' : observable, 'time': alpha[:, 0, 0], 'target': o_target, 'ext_param': alpha[0,0,1]}

  def validation_epoch_end(self, outs):
    fig, ax = plt.subplots(figsize=(10,6))
    for i, out in enumerate(outs):
      ext_param = float(out['ext_param'].cpu())
      ax.plot(out['time'].cpu(), out['target'].cpu(), label=f'target h={ext_param:.1f}', c=f'C{i}', ls='--')
      ax.plot(out['time'].cpu(), out['observable'].cpu(), label=f'model_pred h={ext_param}', c=f'C{i}')
    ax.set_xlabel('ht')
    ax.set_ylabel('$E\,[\sigma_x]$')
    ax.legend()
    ax.set_title('TFI magnetization')
    self.logger.experiment.add_figure('magnetization', fig)
    fig.savefig('magnetization.png')
    plt.close(fig)
  
  def configure_optimizers(self):
    #optimizer = torch.optim.LBFGS(self.parameters(), lr=1e-2)
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #optimizer = torch.optim.RMSprop(self.parameters())
    #optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
    return optimizer


model = Model(lattice_sites, h_mat, h_map, o_mat, o_map, t_min=0, t_max=2)
#print(model)

trainer = pl.Trainer(fast_dev_run=False, gpus=1, max_epochs=30)
trainer.fit(model, train_dataloader, val_dataloader)



