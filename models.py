import tNN
import activations as act
import torch
from torch import nn
import pytorch_lightning as pl

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
        psi_type = torch.complex128
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)