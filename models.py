import tNN
import activations as act
import torch
from torch import nn
import pytorch_lightning as pl
import math
"""
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
"""


class multConv2(tNN.Wave_Fun):
    def __init__(self, lattice_sites, num_h_params, learning_rate):
        super().__init__(lattice_sites = lattice_sites)
        self.save_hyperparameters()
        self.lattice_sites = lattice_sites
        self.lr = learning_rate
        self.opt = torch.optim.Adam
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        conv_out = 32 * ((self.lattice_sites - 3 + 2) + 1)
        mult_size = 256
        self.lattice_net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1, padding_mode='circular'),
            nn.CELU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.CELU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.CELU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(conv_out, mult_size),
        )



        tNN_hidden = 256
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
            act.ComplexLinear( psi_hidden, psi_hidden),
            act.complex_celu(),
            act.ComplexLinear( psi_hidden, psi_hidden),
            act.complex_celu(),
            act.ComplexLinear( psi_hidden, psi_hidden),
            act.complex_celu(),
            act.ComplexLinear( psi_hidden, 1),
        )

    def forward(self, spins, alpha):
        lat_out = self.lattice_net(spins.unsqueeze(1))
        t_out = self.tNN(alpha)
        psi_out = self.psi( (t_out * lat_out) )
        return psi_out

    def configure_optimizers(self):
        optimizer = self.opt(self.parameters(), lr=self.lr)
        lr_scheduler = self.scheduler(optimizer, patience=1, verbose=True, factor=.5)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'val_loss'}

class multConvDeep2(tNN.Wave_Fun):
    def __init__(self, lattice_sites, num_h_params, learning_rate):
        super().__init__(lattice_sites = lattice_sites)
        self.save_hyperparameters()
        self.lattice_sites = lattice_sites
        self.lr = learning_rate
        self.opt = torch.optim.Adam
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        conv_out = 32 * ((self.lattice_sites - 2 + 2) + 2 * 1)
        mult_size = 512
        self.lattice_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=2, padding=1, padding_mode='circular'),
            nn.CELU(),
            nn.Conv1d(16, 32, kernel_size=2, padding=1, padding_mode='zeros'),
            nn.CELU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(conv_out, mult_size),
        )

        tNN_hidden = 128
        self.tNN_first = nn.Sequential(
            nn.Linear(1 + num_h_params, tNN_hidden),
            nn.CELU(),
        )
        self.tNN_hidden1 = nn.Sequential(
            nn.Linear(tNN_hidden, tNN_hidden),
            nn.CELU(),
            nn.Linear(tNN_hidden, tNN_hidden),
            nn.CELU(),
        )
        self.tNN_hidden2 = nn.Sequential(
            nn.Linear( 2 * tNN_hidden, tNN_hidden),
            nn.CELU(),
            nn.Linear(tNN_hidden, tNN_hidden),
            nn.CELU(),
        )

        self.tNN_last = nn.Sequential(
            nn.Linear( 2 * tNN_hidden, mult_size),
        )

        after_act = int( mult_size / 2 )
        psi_hidden = 64
        self.psi_first = nn.Sequential(
            act.rad_phase_act(),
            act.ComplexLinear( after_act, psi_hidden),
            act.complex_celu(),
        )
        self.psi_hidden1 = nn.Sequential(
            act.ComplexLinear( psi_hidden, psi_hidden),
            act.complex_celu(),
            act.ComplexLinear( psi_hidden, psi_hidden),
            act.complex_celu(),
        )            
        self.psi_hidden2 = nn.Sequential(
            act.ComplexLinear( 2 * psi_hidden, psi_hidden),
            act.complex_celu(),
            act.ComplexLinear( psi_hidden, psi_hidden),
            act.complex_celu(),
        )
        self.psi_last = nn.Sequential(
            act.ComplexLinear( 2 * psi_hidden, 1),
        )

    def forward(self, spins, alpha):
        lat_out = self.lattice_net(spins.unsqueeze(1))
        t0 = self.tNN_first(alpha)
        t1 = self.tNN_hidden1(t0)
        t2 = self.tNN_hidden2(torch.cat((t0, t1), dim=1))
        #t2 = self.tNN_hidden2(t0 + t1)
        t_out = self.tNN_last(torch.cat((t0, t2), dim=1))
        #t_out = self.tNN_last(t1 + t2)
        psi_0 = self.psi_first( t_out * lat_out )
        psi_1 = self.psi_hidden1( psi_0 )
        psi_2 = self.psi_hidden2( torch.cat((psi_0, psi_1), dim=1) )
        #psi_2 = self.psi_hidden2( psi_0 + psi_1 )
        psi_out = self.psi_last( torch.cat((psi_0, psi_2), dim=1) )
        #psi_out = self.psi_last( psi_1 + psi_2 )
        return psi_out

    def configure_optimizers(self):
        optimizer = self.opt(self.parameters(), lr=self.lr)
        lr_scheduler = self.scheduler(optimizer, patience=3, verbose=True, factor=.5)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'train_loss'}

class multConvDeep(tNN.Wave_Fun):
    def __init__(self, lattice_sites, num_h_params, learning_rate, patience=0):
        super().__init__(lattice_sites = lattice_sites)
        self.save_hyperparameters()
        self.lattice_sites = lattice_sites
        self.lr = learning_rate
        self.patience = patience
        self.opt = torch.optim.Adam
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        conv_out = 32 * ((self.lattice_sites - 2 + 2) + 2 * 1)
        mult_size = 512
        self.lattice_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=2, padding=1, padding_mode='circular'),
            nn.CELU(),
            nn.Conv1d(16, 32, kernel_size=2, padding=1, padding_mode='zeros'),
            nn.CELU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(conv_out, mult_size),
        )

        tNN_hidden = 64
        self.tNN_first = nn.Sequential(
            nn.Linear(1 + num_h_params, tNN_hidden),
            nn.CELU(),
        )
        self.tNN_hidden1 = nn.Sequential(
            nn.Linear(tNN_hidden, tNN_hidden),
            nn.CELU(),
            nn.Linear(tNN_hidden, tNN_hidden),
            nn.CELU(),
        )
        self.tNN_hidden2 = nn.Sequential(
            nn.Linear(2 * tNN_hidden, tNN_hidden),
            nn.CELU(),
            nn.Linear(tNN_hidden, tNN_hidden),
            nn.CELU(),
        )

        self.tNN_last = nn.Sequential(
            nn.Linear(3 * tNN_hidden, mult_size),
        )

        after_act = int( mult_size / 2 )
        psi_hidden = 64
        self.psi_first = nn.Sequential(
            act.rad_phase_act(),
            act.ComplexLinear( after_act, psi_hidden),
            act.complex_celu(),
        )
        self.psi_hidden1 = nn.Sequential(
            act.ComplexLinear( psi_hidden, psi_hidden),
            act.complex_celu(),
            act.ComplexLinear( psi_hidden, psi_hidden),
            act.complex_celu(),
        )            
        self.psi_hidden2 = nn.Sequential(
            act.ComplexLinear( 2 * psi_hidden, psi_hidden),
            act.complex_celu(),
            act.ComplexLinear( psi_hidden, psi_hidden),
            act.complex_celu(),
        )
        self.psi_last = nn.Sequential(
            act.ComplexLinear( 3 * psi_hidden, 1),
        )

    def forward(self, spins, alpha):
        lat_out = self.lattice_net(spins.unsqueeze(1))
        t0 = self.tNN_first(alpha)
        t1 = self.tNN_hidden1(t0)
        t2 = self.tNN_hidden2(torch.cat((t0, t1), dim=1))
        #t2 = self.tNN_hidden2(t0 * t1)
        t_out = self.tNN_last(torch.cat((t0, t1, t2), dim=1))
        #t_out = self.tNN_last(t1 * t2)
        psi_0 = self.psi_first( t_out * lat_out )
        psi_1 = self.psi_hidden1( psi_0 )
        psi_2 = self.psi_hidden2( torch.cat((psi_0, psi_1), dim=1) )
        psi_out = self.psi_last( torch.cat((psi_0, psi_1, psi_2), dim=1) )
        return psi_out

    def configure_optimizers(self):
        optimizer = self.opt(self.parameters(), lr=self.lr)
        lr_scheduler = self.scheduler(optimizer, patience=self.patience, verbose=True, factor=.5)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'train_loss'}


class init_fixed(tNN.Wave_Fun):
    def __init__(self, lattice_sites, num_h_params, learning_rate, psi_init, patience=0):
        super().__init__(lattice_sites = lattice_sites)
        self.save_hyperparameters()
        self.lattice_sites = lattice_sites
        self.psi_init = psi_init
        self.lr = learning_rate
        self.patience = patience
        self.opt = torch.optim.Adam
        self.scheduler = torch.optim.lr_scheduler.StepLR
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        conv_out = 32 * ((self.lattice_sites - 2 + 2) + 2 * 1)
        mult_size = 512
        self.lattice_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=2, padding=1, padding_mode='circular'),
            nn.CELU(),
            nn.Conv1d(16, 32, kernel_size=2, padding=1, padding_mode='zeros'),
            nn.CELU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(conv_out, mult_size),
        )

        tNN_hidden = 64
        self.tNN_first = nn.Sequential(
            nn.Linear(1 + num_h_params, tNN_hidden),
            nn.CELU(),
        )
        self.tNN_hidden1 = nn.Sequential(
            nn.Linear(tNN_hidden, tNN_hidden),
            nn.CELU(),
            nn.Linear(tNN_hidden, tNN_hidden),
            nn.CELU(),
        )
        self.tNN_hidden2 = nn.Sequential(
            nn.Linear(2 * tNN_hidden, tNN_hidden),
            nn.CELU(),
            nn.Linear(tNN_hidden, tNN_hidden),
            nn.CELU(),
        )

        self.tNN_last = nn.Sequential(
            nn.Linear(3 * tNN_hidden, mult_size),
        )

        after_act = int( mult_size / 2 )
        psi_hidden = 64
        self.psi_first = nn.Sequential(
            act.rad_phase_act(),
            act.ComplexLinear( after_act, psi_hidden),
            act.complex_celu(),
        )
        self.psi_hidden1 = nn.Sequential(
            act.ComplexLinear( psi_hidden, psi_hidden),
            act.complex_celu(),
            act.ComplexLinear( psi_hidden, psi_hidden),
            act.complex_celu(),
        )            
        self.psi_hidden2 = nn.Sequential(
            act.ComplexLinear( 2 * psi_hidden, psi_hidden),
            act.complex_celu(),
            act.ComplexLinear( psi_hidden, psi_hidden),
            act.complex_celu(),
        )
        self.psi_last = nn.Sequential(
            act.ComplexLinear( 3 * psi_hidden, 1),
        )

    def forward(self, spins, alpha):
        lat_out = self.lattice_net(spins.unsqueeze(1))
        t0 = self.tNN_first(alpha)
        t1 = self.tNN_hidden1(t0)
        t2 = self.tNN_hidden2(torch.cat((t0, t1), dim=1))
        #t2 = self.tNN_hidden2(t0 * t1)
        t_out = self.tNN_last(torch.cat((t0, t1, t2), dim=1))
        #t_out = self.tNN_last(t1 * t2)
        psi_0 = self.psi_first( t_out * lat_out )
        psi_1 = self.psi_hidden1( psi_0 )
        psi_2 = self.psi_hidden2( torch.cat((psi_0, psi_1), dim=1) )
        psi_out = self.psi_last( torch.cat((psi_0, psi_1, psi_2), dim=1) )
        return (1 - torch.exp(- alpha[:, :1]) ) * psi_out + self.psi_init(spins, self.lattice_sites)

    def configure_optimizers(self):
        optimizer = self.opt(self.parameters(), lr=self.lr)
        #lr_scheduler = self.scheduler(optimizer, patience=self.patience, verbose=True, factor=.5)
        lr_scheduler = self.scheduler(optimizer, step_size=1, verbose=True, gamma=0.5)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'train_loss'}


class time_transformer(tNN.Wave_Fun):
    def __init__(self, lattice_sites, num_h_params, learning_rate, psi_init, patience=0):
        super().__init__(lattice_sites = lattice_sites)
        self.save_hyperparameters()
        self.lattice_sites = lattice_sites
        self.psi_init = psi_init
        self.lr = learning_rate
        self.patience = patience
        self.opt = torch.optim.Adam
        self.scheduler = torch.optim.lr_scheduler.StepLR
        conv_out = 32 * 6
        mult_size = 128
        self.lattice_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=2, padding=1, padding_mode='circular'),
            nn.CELU(),
            nn.Conv1d(16, 32, kernel_size=2, padding=1, padding_mode='zeros'),
            nn.CELU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(conv_out, mult_size),
        )

        tNN_hidden = 64
        self.tNN_first = nn.Sequential(
            nn.Linear(1 + num_h_params, tNN_hidden),
            nn.CELU(),
        )
        self.tNN_hidden1 = nn.Sequential(
            nn.Linear(tNN_hidden, tNN_hidden),
            nn.CELU(),
            nn.Linear(tNN_hidden, tNN_hidden),
            nn.CELU(),
        )
        self.tNN_hidden2 = nn.Sequential(
            nn.Linear(2 * tNN_hidden, tNN_hidden),
            nn.CELU(),
            nn.Linear(tNN_hidden, tNN_hidden),
            nn.CELU(),
        )

        self.tNN_last = nn.Sequential(
            nn.Linear(3 * tNN_hidden, mult_size),
        )

        self.attention = act.time_attention(mult_size, 8)
        self.flatten_conv = nn.Flatten(start_dim=1, end_dim=-1)

        after_act = int( mult_size / 2 )
        psi_hidden = 64
        self.psi_first = nn.Sequential(
            act.rad_phase_act(),
            act.ComplexLinear( after_act, psi_hidden),
            act.complex_celu(),
        )
        self.psi_hidden1 = nn.Sequential(
            act.ComplexLinear( psi_hidden, psi_hidden),
            act.complex_celu(),
            act.ComplexLinear( psi_hidden, psi_hidden),
            act.complex_celu(),
        )            
        self.psi_hidden2 = nn.Sequential(
            act.ComplexLinear( 2 * psi_hidden, psi_hidden),
            act.complex_celu(),
            act.ComplexLinear( psi_hidden, psi_hidden),
            act.complex_celu(),
        )
        self.psi_last = nn.Sequential(
            act.ComplexLinear( 3 * psi_hidden, 1),
        )

    def forward(self, spins, alpha):
        lat_out = self.lattice_net(spins.unsqueeze(1))
        t0 = self.tNN_first(alpha)
        t1 = self.tNN_hidden1(t0)
        t2 = self.tNN_hidden2(torch.cat((t0, t1), dim=1))
        #t2 = self.tNN_hidden2(t0 * t1)
        t_out = self.tNN_last(torch.cat((t0, t1, t2), dim=1))
        att_out = self.attention(t_out, lat_out)

        psi_0 = self.psi_first( self.flatten_conv(att_out) )
        psi_1 = self.psi_hidden1( psi_0 )
        psi_2 = self.psi_hidden2( torch.cat((psi_0, psi_1), dim=1) )
        psi_out = self.psi_last( torch.cat((psi_0, psi_1, psi_2), dim=1) )
        return (1 - torch.exp(- alpha[:, :1]) ) * psi_out + self.psi_init(spins, self.lattice_sites)

    def configure_optimizers(self):
        optimizer = self.opt(self.parameters(), lr=self.lr)
        lr_scheduler = self.scheduler(optimizer, step_size=1, verbose=True, gamma=0.5)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'train_loss'}