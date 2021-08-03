import tNN
import activations as act
import torch
from torch import nn
import pytorch_lightning as pl
import math

@tNN.wave_function
class simpleModel(pl.LightningModule):
    def __init__(self, lattice_sites, num_h_params, learning_rate):
        super().__init__()
        self.lattice_sites = lattice_sites
        self.lr = learning_rate
        hidden_size = 64
        net_dtype = torch.complex128
        self.simple_net = nn.Sequential(
            act.to_complex(),
            nn.Linear( lattice_sites + num_h_params + 1, hidden_size, dtype=net_dtype),
            act.complex_celu(),
            nn.Linear( hidden_size, hidden_size, dtype=net_dtype),
            act.complex_celu(),
            nn.Linear( hidden_size, hidden_size, dtype=net_dtype),
            act.complex_celu(),
            nn.Linear( hidden_size, 1, dtype=net_dtype)
        )

    def forward(self, spins, alpha):
        return self.simple_net(torch.cat((spins, alpha), dim=1))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

@tNN.wave_function
class multModel(pl.LightningModule):
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

@tNN.wave_function
class catModel(pl.LightningModule):
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

        psi_hidden = mult_size
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
        psi_out = self.psi( torch.cat((t_out, lat_out), dim=1) )
        return psi_out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

@tNN.wave_function
class multConvModel(pl.LightningModule):
    def __init__(self, lattice_sites, num_h_params, learning_rate):
        super().__init__()
        self.lattice_sites = lattice_sites
        self.lr = learning_rate
        self.opt = torch.optim.Adam
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        conv_out = 16 * (lattice_sites + 1)
        mult_size = 128
        self.lattice_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=2, padding=1, padding_mode='circular'),
            nn.CELU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(conv_out, mult_size),
            #nn.CELU()
        )

        tNN_hidden = 128
        self.tNN = nn.Sequential(
            nn.Linear(1 + num_h_params, 64),
            nn.CELU(),
            nn.Linear(64, tNN_hidden),
            nn.CELU(),
            nn.Linear(tNN_hidden, tNN_hidden),
            nn.CELU(),
            nn.Linear(tNN_hidden, mult_size),
            #nn.CELU()
        )

        psi_hidden = int( mult_size / 2 )
        psi_type = torch.complex64
        self.psi = nn.Sequential(
            nn.Linear(mult_size, mult_size),
            act.rad_phase_act(),
            nn.Linear( psi_hidden, psi_hidden, dtype=psi_type),
            act.complex_celu(),
            nn.Linear( psi_hidden, psi_hidden, dtype=psi_type),
            act.complex_celu(),
            nn.Linear( psi_hidden, 1, dtype=psi_type),
        )

    def forward(self, spins, alpha):
        lat_out = self.lattice_net(spins.unsqueeze(1))
        t_out = self.tNN(alpha)
        psi_out = self.psi( (t_out * lat_out) )
        return psi_out

    def configure_optimizers(self):
        optimizer = self.opt(self.parameters(), lr=self.lr)
        #lr_scheduler = self.scheduler(optimizer, patience=0)
        return {'optimizer': optimizer}#, 'lr_scheduler': lr_scheduler, 'monitor': 'val_loss'}


@tNN.wave_function
class realModel(pl.LightningModule):
    def __init__(self, lattice_sites, num_h_params, learning_rate):
        super().__init__()
        self.lattice_sites = lattice_sites
        self.lr = learning_rate
        self.opt = torch.optim.Adam
        conv_out = 16 * (lattice_sites + 1)
        mult_size = 128
        self.lattice_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=2, padding=1, padding_mode='circular'),
            nn.CELU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(conv_out, mult_size),
            #nn.CELU()
        )

        tNN_hidden = 128
        self.tNN = nn.Sequential(
            nn.Linear(1 + num_h_params, 64),
            nn.CELU(),
            nn.Linear(64, tNN_hidden),
            nn.CELU(),
            nn.Linear(tNN_hidden, tNN_hidden),
            nn.CELU(),
            nn.Linear(tNN_hidden, mult_size),
            #nn.CELU()
        )

        psi_hidden = 64
        self.psi = nn.Sequential(
            nn.Linear(mult_size, psi_hidden),
            nn.CELU(),
            nn.Linear( psi_hidden, psi_hidden),
            nn.CELU(),
            nn.Linear( psi_hidden, psi_hidden),
            nn.CELU(),
            nn.Linear( psi_hidden, 2),
            act.rad_phase_act()
        )

    def forward(self, spins, alpha):
        lat_out = self.lattice_net(spins.unsqueeze(1))
        t_out = self.tNN(alpha)
        psi_out = self.psi( (t_out * lat_out) )
        return psi_out

    def configure_optimizers(self):
        return self.opt(self.parameters(), lr=self.lr)


@tNN.wave_function
class multConv2(pl.LightningModule):
    def __init__(self, lattice_sites, num_h_params, learning_rate):
        super().__init__()
        self.lattice_sites = lattice_sites
        self.lr = learning_rate
        self.opt = torch.optim.Adam
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        conv_out = 32 * (lattice_sites)
        mult_size = 256
        self.lattice_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=2, padding=1, padding_mode='circular'),
            nn.CELU(),
            nn.Conv1d(16, 32, kernel_size=2),
            nn.CELU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(conv_out, mult_size),
        )

        tNN_hidden = 128
        self.tNN = nn.Sequential(
            nn.Linear(1 + num_h_params, 64),
            nn.CELU(),
            nn.Linear(64, tNN_hidden),
            nn.CELU(),
            nn.Linear(tNN_hidden, tNN_hidden),
            nn.CELU(),
            nn.Linear(tNN_hidden, mult_size),
        )
        after_act = int( mult_size / 2 )
        psi_hidden = 128
        psi_type = torch.complex64
        self.psi = nn.Sequential(
            act.rad_phase_act(),
            nn.Linear( after_act, psi_hidden, dtype=psi_type),
            act.complex_celu(),
            nn.Linear( psi_hidden, psi_hidden, dtype=psi_type),
            act.complex_celu(),
            nn.Linear( psi_hidden, 64, dtype=psi_type),
            act.complex_celu(),
            nn.Linear( 64, 1, dtype=psi_type),
        )

    def forward(self, spins, alpha):
        lat_out = self.lattice_net(spins.unsqueeze(1))
        t_out = self.tNN(alpha)
        psi_out = self.psi( (t_out * lat_out) )
        return psi_out

    def configure_optimizers(self):
        optimizer = self.opt(self.parameters(), lr=self.lr)
        lr_scheduler = self.scheduler(optimizer, patience=0, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'train_loss'}


@tNN.wave_function
class tryout(pl.LightningModule):
    def __init__(self, lattice_sites, num_h_params, learning_rate):
        super().__init__()
        self.lattice_sites = lattice_sites
        self.lr = learning_rate
        self.opt = torch.optim.Adam
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        self.lat_out_shape = (-1, 32, (lattice_sites + 3))
        mult_size = 32 * (lattice_sites + 3)
        self.lattice_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=2, padding=1, padding_mode='circular'),
            nn.CELU(),
            nn.Conv1d(16, 32, kernel_size=2, padding=1),
            nn.CELU(),
            nn.Conv1d(32, 32, kernel_size=2, padding=1),
        )
        tNN_hidden = 128
        self.tNN = nn.Sequential(
            nn.Linear(1 + num_h_params, 64),
            nn.CELU(),
            nn.Linear(64, tNN_hidden),
            nn.CELU(),
            nn.Linear(tNN_hidden, tNN_hidden),
            nn.CELU(),
            nn.Linear(tNN_hidden, mult_size),
        )
        after_act = int( mult_size / 2 )
        psi_hidden = 128
        psi_type = torch.complex64
        self.psi = nn.Sequential(
            act.rad_phase_conv_act(),
            act.ComplexConv1d(int(self.lat_out_shape[1]/2), 32, kernel_size=2, padding=1),
            act.complex_celu(),
            act.ComplexConv1d(32 , 32, kernel_size=2, padding=1),
            act.complex_celu(),
            act.ComplexConv1d(32 , 32, kernel_size=2, padding=1),
            act.complex_celu(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(32 * (lattice_sites + 6), 1, dtype=psi_type)
        )

    def forward(self, spins, alpha):
        lat_out = self.lattice_net(spins.unsqueeze(1))
        t_out = self.tNN(alpha).unsqueeze(2)
        #print('lat_out, t_out shape', lat_out.shape, t_out.shape)
        t_out = t_out.reshape(self.lat_out_shape)
        #print('lat_out, t_out shape', lat_out.shape, t_out.shape)
        psi_out = self.psi( (t_out * lat_out) )
        #print('psi_out_shape', psi_out.shape)
        return psi_out

    def configure_optimizers(self):
        optimizer = self.opt(self.parameters(), lr=self.lr)
        lr_scheduler = self.scheduler(optimizer, patience=0, verbose=True, factor=.5)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'val_loss'}