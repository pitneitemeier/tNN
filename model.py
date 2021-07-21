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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
g_dtype = torch.float64



###setting up hamiltonian ###
lattice_sites = 4

h2_range = [(0.7, 1.1)]

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

ED_data_02 = np.loadtxt('ED_data/ED_data4_02.csv', delimiter=',')
ED_data_05 = np.loadtxt('ED_data/ED_data4_05.csv', delimiter=',')
ED_data_07 = np.loadtxt('ED_data/ED_data4_07.csv', delimiter=',')
ED_data_10 = np.loadtxt('ED_data/ED_data4_1.csv', delimiter=',')
ED_data_13 = np.loadtxt('ED_data/ED_data4_13.csv', delimiter=',')

#ED_data = np.stack((ED_data_02, ED_data_05, ED_data_07, ED_data_10, ED_data_13))
#ext_params = np.array([0.2, 0.5, 0.7, 1, 1.3]).reshape(5,1)
ED_data = np.stack((ED_data_10, ED_data_07))
ext_params = np.array([1.0, 0.7]).reshape(2,1)
val_data = Datasets.Val_Data(lattice_sites, ED_data, o_mat, g_dtype, ext_params)
val_dataloader = DataLoader(val_data, batch_size=1, num_workers=24)
val_iter = iter(val_dataloader)

train_data = Datasets.Train_Data(lattice_sites, [h1_mat, h2_mat], h2_range, o_mat, g_dtype)
train_dataloader = DataLoader(dataset=train_data, batch_size=200, num_workers=24)
data_iter = iter(train_dataloader)


class Model(pl.LightningModule):

  def __init__(self, lattice_sites, h_map, o_init_map, t_min, t_max):
    '''
    Initializer for neural net
    Parameters
    __________
    lattice_sites: int
    h_mat_list: tensor, dtype=complex
      the matrix elements of the hamiltonian 
      shape = (num_ext_params, )
    '''
    super().__init__()
    
    '''
    self.lattice_net = nn.Sequential(
      nn.Conv1d(1, 8, kernel_size=2, padding=1, padding_mode='circular'),
      utils.even_act(),
      nn.Conv1d(8, 8, kernel_size=2, padding=1, padding_mode='zeros'),
      nn.CELU(),
      nn.Conv1d(8, 16, kernel_size=2, padding=1, padding_mode='zeros'),
      nn.CELU(),
      nn.Flatten(start_dim=1, end_dim=-1),
    )
    '''
    self.t_min = t_min
    self.t_max = t_max
    lattice_hidden = 64
    #mult_size = 16 * (lattice_sites + 3)
    mult_size = 64
    tNN_hidden = 64
    
    
    
    self.lattice_net = nn.Sequential(
      nn.Linear( lattice_sites, lattice_hidden, dtype=g_dtype),
      utils.even_act(),
      nn.Linear( lattice_hidden, mult_size, dtype=g_dtype)
    )
    
    self.tNN = nn.Sequential(
      nn.Linear(2, tNN_hidden, dtype=g_dtype),
      nn.CELU(),
      nn.Linear(tNN_hidden, tNN_hidden, dtype=g_dtype),
      nn.CELU(),
      nn.Linear(tNN_hidden, mult_size, dtype=g_dtype),
      nn.CELU()
    )
    '''
    self.tNN = nn.Sequential(
      utils.to_complex(),
      nn.Linear(2, 16, dtype=torch.complex64),
      utils.complex_CELU(),
      nn.Linear(16, mult_size, dtype=torch.complex64),
      utils.complex_CELU(),
    ) 
    psi_hidden = 64
    self.psi = nn.Sequential(
      nn.Linear( mult_size, psi_hidden ),
      nn.CELU(),
      nn.Linear(psi_hidden, psi_hidden),
      nn.CELU(),
      nn.Linear(psi_hidden, psi_hidden),
      nn.CELU(),
      nn.Linear(psi_hidden, 2)
    )
    '''
    psi_hidden = 64
    psi_type = torch.complex128
    self.psi = nn.Sequential(
      utils.to_complex(),
      nn.Linear( mult_size, psi_hidden, dtype=psi_type),
      utils.odd_act(),
      nn.Linear( psi_hidden, psi_hidden, dtype=psi_type),
      utils.odd_act(),
      nn.Linear( psi_hidden, 1, dtype=psi_type),
      utils.odd_act(),

    )
    

    '''
    hidden_size = 64
    self.simple_model = nn.Sequential(
      nn.Linear(  lattice_sites + 2 , hidden_size, dtype=g_dtype),
      nn.ReLU(),
      utils.to_complex(),
      nn.Linear( hidden_size, hidden_size, dtype=torch.complex64),
      utils.complex_tanh(),
      nn.Linear( hidden_size, hidden_size, dtype=torch.complex64),
      utils.complex_tanh(),
      nn.Linear( hidden_size, 1, dtype=torch.complex64),
    )
    '''
    self.flatten_spins = nn.Flatten(end_dim=-1)
    self.h_map = h_map.to(device)
    self.o_init_map = o_init_map.to(device)

  def forward(self, spins, alpha):
    #unsqueeze since circular padding needs tensor of dim 3
    #lat_out = self.lattice_net(spins.unsqueeze(1))
    #one_hot_spins = utils.flat_one_hot(spins)
    #one_hot_spins = self.flatten_spins(one_hot_spins)
    #lat_out = self.lattice_net(one_hot_spins)
    
    lat_out = self.lattice_net(spins)
    
    t_out = self.tNN(alpha)

    psi = self.psi( (t_out * lat_out) )
    #psi = self.psi( torch.cat((t_out, lat_out), dim=1) )

    #rad_and_phase = self.psi( torch.cat((t_out, lat_out), dim=1))
    #rad_and_phase = self.psi( t_out * lat_out )
    #psi = rad_and_phase[:, 0] * torch.exp( np.pi * 2j * self.sigmoid(rad_and_phase[:, 1]))
    #psi = rad_and_phase[:, 0] * torch.exp( 1.j * rad_and_phase[:, 1] ) +  rad_and_phase[:, 2] * torch.exp( -1.j * rad_and_phase[:, 3] )
    #psi = self.simple_model( torch.cat((spins, alpha), dim=1))
    return psi
    
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
    spins, alpha, alpha_0, h_mat, o_mat = batch
    n = self.current_epoch % 5 + 1
    N = 100
    t_max = self.t_min + (self.t_max - self.t_min) * np.log(10 * n / N + 1) / np.log( 11 )
    loss_weight = 1e-2 / (t_max/self.t_max + 1e-2)
    alpha[:, :, 0] = (t_max - self.t_min) * alpha[:, :, 0] + self.t_min
    alpha.requires_grad = True
    self.log('end time', t_max, prog_bar=True)
    
    #get psi(s, alpha)
    psi_s = self.call_forward(spins, alpha)
    #get s' and psi(s', alpha) for h
    sp_h = utils.get_sp(spins, self.h_map)
    psi_sp_h = self.call_forward_sp(sp_h, alpha)
    #calc h_loc for h
    h_loc = utils.calc_Oloc(psi_sp_h, h_mat, spins)
    dt_psi_s = utils.calc_dt_psi(psi_s, alpha)
    
    #get s' and psi(s', alpha) for o at t=0
    sp_o = utils.get_sp(spins, self.o_init_map)
    psi_sp_o = self.call_forward_sp(sp_o, alpha_0)
    psi_s_0 = self.call_forward(spins, alpha_0)
    
    #calc o_loc for o
    o_loc = utils.calc_Oloc(psi_sp_o, o_mat, spins)

    #calc loss
    loss, schrodinger, init_cond, norm = utils.train_loss2(dt_psi_s, h_loc, psi_s, psi_s_0, o_loc, alpha, loss_weight)
    self.log('schrodinger', schrodinger, prog_bar=True, logger=True)
    self.log('init_cond', init_cond, prog_bar=True, logger=True)
    self.log('norm', norm, prog_bar=True, logger=True)

    return {'loss': loss}

  def validation_step(self, batch, batch_idx):
    spins, alpha, o_mat, o_target = batch
    
    psi_s = self.call_forward(spins.squeeze(0), alpha.squeeze(0))
    sp_o = utils.get_sp(spins.squeeze(0), self.o_init_map)
    psi_sp_o = self.call_forward_sp(sp_o, alpha.squeeze(0))
    o_loc = utils.calc_Oloc(psi_sp_o, o_mat.squeeze(0), spins.squeeze(0))
    val_loss, observable = utils.val_loss(psi_s, o_loc, o_target.squeeze(0))
    #self.log('val_loss', val_loss, prog_bar=True, logger=True)

    '''
    fig, ax = plt.subplots()
    ax.plot(alpha[:, 0, 0].cpu(), observable.cpu(), label='model prediction')
    ax.plot(alpha[:, 0, 0].cpu(), o_target.cpu(), label='ED Result', ls='--')
    ax.legend()
    #self.logger.experiment.add_figure('magnetization', fig)
    fig.savefig('magnetization.png')
    plt.close(fig)
    '''
    #print('obs_shape', observable.shape)
    #print('alpha_shape', alpha.shape)
    #print('o_target shape', o_target.shape)
    return {'observable' : observable, 'time': alpha[0, :, 0, 0], 'target': o_target.squeeze(0), 'ext_param': alpha[0,0,0,1]}
  
  def validation_epoch_end(self, outs):
    fig, ax = plt.subplots()
    for i, out in enumerate(outs):
      ext_param = float(out['ext_param'].cpu())
      ax.plot(out['time'].cpu(), out['target'].cpu(), label=f'target g={ext_param:.1f}', c=f'C{i}', ls='--')
      ax.plot(out['time'].cpu(), out['observable'].cpu(), label=f'model_pred g={ext_param}', c=f'C{i}')
    ax.legend()
    fig.savefig('magnetization.png')
    plt.close(fig)

  def configure_optimizers(self):
    #optimizer = torch.optim.LBFGS(self.parameters(), lr=1e-2)
    optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
    #optimizer = torch.optim.RMSprop(self.parameters())
    #optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
    return optimizer



model = Model(lattice_sites, h_map, o_map, t_min=0, t_max=1.5)
print(model)

trainer = pl.Trainer(fast_dev_run=False, gpus=1)
trainer.fit(model, train_dataloader, val_dataloader)

