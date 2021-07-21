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
g_dtype = torch.float32



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
ED_data = np.loadtxt('ED_data-train4.csv', delimiter=',')

val_data = Datasets.Val_Data(lattice_sites, ED_data, (0.5,), o_mat, g_dtype)
#val_data.to_device(device)
val_dataloader = DataLoader(val_data, batch_size=len(val_data), num_workers=24)
val_iter = iter(val_dataloader)

'''
train_data = Datasets.Train_Data(lattice_sites, [h1_mat, h2_mat], h2_range, o_mat, g_dtype,  t_max=3)
#train_data.to_device(device)
train_dataloader = DataLoader(dataset=train_data, batch_size=500, num_workers=24)
data_iter = iter(train_dataloader)
'''

class Model(pl.LightningModule):

  def __init__(self, lattice_sites, h_map, o_init_map):
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
    
    lattice_hidden = 64
    #mult_size = 16 * (lattice_sites + 3)
    mult_size = 64
    tNN_hidden = 64
    psi_hidden = 64
    
    
    self.lattice_net = nn.Sequential(
      nn.Linear( lattice_sites, lattice_hidden),
      nn.CELU(),
      nn.Linear(lattice_hidden, mult_size),
      nn.CELU()
    )
    
    self.tNN = nn.Sequential(
      nn.Linear(2, 32),
      nn.CELU(),
      nn.Linear(32, tNN_hidden),
      nn.CELU(),
      nn.Linear(tNN_hidden, mult_size),
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

    self.psi = nn.Sequential(
      nn.Linear( mult_size, psi_hidden),
      nn.CELU(),
      nn.Linear(psi_hidden, psi_hidden),
      nn.CELU(),
      utils.Euler_act(),
      nn.Linear(int(psi_hidden / 2), 1, dtype=torch.complex64)
    )
    
    '''
    
    hidden_size = 64
    self.simple_model = nn.Sequential(
      nn.Linear( 2 * lattice_sites + 2 , hidden_size, dtype=g_dtype),
      nn.CELU(),
      nn.Linear( hidden_size, hidden_size, dtype=g_dtype),
      nn.CELU(),
      nn.Linear( hidden_size, hidden_size, dtype=g_dtype),
      nn.CELU(),
      nn.Linear( hidden_size, 4, dtype=g_dtype)
    )
    '''
    self.flatten_spins = nn.Flatten(end_dim=-1)
    self.h_map = h_map.to(device)
    self.o_init_map = o_init_map.to(device)
    self.sigmoid = nn.Sigmoid()

  def forward(self, spins, alpha):
    #unsqueeze since circular padding needs tensor of dim 3
    #lat_out = self.lattice_net(spins.unsqueeze(1))
    #one_hot_spins = utils.flat_one_hot(spins)
    #one_hot_spins = self.flatten_spins(one_hot_spins)
    #lat_out = self.lattice_net(one_hot_spins)
    
    lat_out = self.lattice_net(spins)
    
    t_out = self.tNN((alpha - 1.5) / 3)

    psi = self.psi( (t_out * lat_out) )

    #rad_and_phase = self.psi( torch.cat((t_out, lat_out), dim=1))
    #rad_and_phase = self.psi( t_out * lat_out )
    #psi = rad_and_phase[:, 0] * torch.exp( np.pi * 2j * self.sigmoid(rad_and_phase[:, 1]))
    #psi = rad_and_phase[:, 0] * torch.exp( 1.j * rad_and_phase[:, 1] ) +  rad_and_phase[:, 2] * torch.exp( -1.j * rad_and_phase[:, 3] )

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
    spins, alpha, o_mat, o_target = batch
    
    psi_s = self.call_forward(spins, alpha)
    sp_o = utils.get_sp(spins, self.o_init_map)
    psi_sp_o = self.call_forward_sp(sp_o, alpha)
    o_loc = utils.calc_Oloc(psi_sp_o, o_mat, spins)
    loss, observable = utils.val_loss(psi_s, o_loc, o_target)

    return {'loss': loss}

  def validation_step(self, batch, batch_idx):
    spins, alpha, o_mat, o_target = batch
    
    psi_s = self.call_forward(spins, alpha)
    sp_o = utils.get_sp(spins, self.o_init_map)
    psi_sp_o = self.call_forward_sp(sp_o, alpha)
    o_loc = utils.calc_Oloc(psi_sp_o, o_mat, spins)
    val_loss, observable = utils.val_loss(psi_s, o_loc, o_target)
    self.log('val_loss', val_loss, prog_bar=True, logger=True)
    fig, ax = plt.subplots()
    ax.plot(alpha[:, 0, 0].cpu(), observable.cpu(), label='model prediction')
    ax.plot(alpha[:, 0, 0].cpu(), o_target.cpu(), label='ED Result', ls='--')
    ax.legend()
    #self.logger.experiment.add_figure('magnetization', fig)
    fig.savefig('magnetization.png')
    plt.close(fig)
    return {'val_loss': val_loss}

  def configure_optimizers(self):
    #optimizer = torch.optim.LBFGS(self.parameters())
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #optimizer = torch.optim.RMSprop(self.parameters())
    #optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
    return optimizer



model = Model(lattice_sites, h_map, o_map)
print(model)

trainer = pl.Trainer(fast_dev_run=False, gpus=1)
trainer.fit(model, val_dataloader, val_dataloader)

