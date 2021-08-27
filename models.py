import tNN
import activations as act
import torch
from torch import nn
import pytorch_lightning as pl
import math

class parametrized(tNN.Wave_Fun):
    def __init__(self, lattice_sites, num_h_params, learning_rate, psi_init, act_fun, kernel_size, num_conv_layers, num_conv_features, tNN_hidden, tNN_num_hidden, mult_size, psi_hidden, psi_num_hidden, init_decay=1, patience=0, optimizer=torch.optim.Adam):
        super().__init__(lattice_sites = lattice_sites, name='parametrized')
        if mult_size % 2 != 0:
            mult_size+=1
        self.save_hyperparameters()
        self.lattice_sites = lattice_sites
        self.psi_init = psi_init
        self.init_decay = init_decay
        self.lr = learning_rate
        self.patience = patience
        self.opt = optimizer
        self.scheduler = torch.optim.lr_scheduler.StepLR
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

        self.lattice_net = act.lattice_block(kernel_size, mult_size, num_conv_features, num_conv_layers, act_fun, self.lattice_sites)
        self.tNN = act.selfAttention(input_size=1+num_h_params, output_size=mult_size, hidden_size=tNN_hidden, num_hidden_layers=tNN_num_hidden, act_fun=act_fun)
        self.combination_block = act.combination_block(mult_size, psi_hidden, act.complex_celu)
        self.psi = act.ComplexSelfAttention(psi_hidden, 1, psi_hidden, psi_num_hidden, act.complex_celu)

    def forward(self, spins, alpha):
        lat_out = self.lattice_net(spins.unsqueeze(1))
        t_out = self.tNN(alpha)
        comb = self.combination_block(lat_out, t_out)
        psi_out = self.psi(comb)
        return (1 - torch.exp(- self.init_decay * alpha[:, :1]) ) * psi_out + self.psi_init(spins, self.lattice_sites)

    def configure_optimizers(self):
        optimizer = self.opt(self.parameters(), lr=self.lr)
        #lr_scheduler = self.scheduler(optimizer, patience=self.patience, verbose=True, factor=.5)
        lr_scheduler = self.scheduler(optimizer, step_size=2, verbose=True, gamma=0.5)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'train_loss'}

class ParametrizedFeedForward(tNN.Wave_Fun):
    def __init__(self, lattice_sites, num_h_params, learning_rate, psi_init, act_fun, kernel_size, 
            num_conv_layers, num_conv_features, tNN_hidden, tNN_num_hidden, mult_size, psi_hidden, psi_num_hidden, 
            init_decay=1, step_size=1, gamma=0.1, optimizer=torch.optim.Adam):
        super().__init__(lattice_sites = lattice_sites, name='FF')
        if mult_size % 2 != 0:
            mult_size+=1
        self.save_hyperparameters()
        self.step_size = step_size
        self.gamma = gamma
        self.lattice_sites = lattice_sites
        self.psi_init = psi_init
        self.init_decay = init_decay
        self.lr = learning_rate
        self.opt = optimizer

        self.lattice_net = act.lattice_block(kernel_size, mult_size, num_conv_features, num_conv_layers, act_fun, self.lattice_sites)
        self.tNN = act.FeedForward(input_size=1+num_h_params, output_size=mult_size, hidden_size=tNN_hidden, num_hidden_layers=tNN_num_hidden, act_fun=act_fun)
        self.combination_block = act.combination_block(mult_size, psi_hidden, act.complex_gelu)
        self.psi = act.ComplexFeedForward(psi_hidden, 1, psi_hidden, psi_num_hidden, act.complex_gelu)

    def forward(self, spins, alpha):
        lat_out = self.lattice_net(spins.unsqueeze(1))
        t_out = self.tNN(alpha)
        comb = self.combination_block(lat_out, t_out)
        psi_out = self.psi(comb)
        return (1 - torch.exp(- self.init_decay * alpha[:, :1]) ) * psi_out + self.psi_init(spins, self.lattice_sites)

    def configure_optimizers(self):
        optimizer = self.opt(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, verbose=True, gamma=self.gamma)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'train_loss'}

class ParametrizedSelfAttention(tNN.Wave_Fun):
    def __init__(self, lattice_sites, num_h_params, learning_rate, psi_init, act_fun, kernel_size, num_conv_layers, num_conv_features, tNN_hidden, tNN_num_hidden, mult_size, psi_hidden, psi_num_hidden, init_decay=1, patience=0, optimizer=torch.optim.Adam):
        super().__init__(lattice_sites = lattice_sites, name='SA')
        if mult_size % 2 != 0:
            mult_size+=1
        self.save_hyperparameters()
        self.lattice_sites = lattice_sites
        self.psi_init = psi_init
        self.init_decay = init_decay
        self.lr = learning_rate
        self.patience = patience
        self.opt = optimizer
        self.scheduler = torch.optim.lr_scheduler.StepLR
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

        self.lattice_net = act.lattice_block(kernel_size, mult_size, num_conv_features, num_conv_layers, act_fun, self.lattice_sites)
        self.tNN = act.selfAttention1(input_size=1+num_h_params, output_size=mult_size, hidden_size=tNN_hidden, num_hidden_layers=tNN_num_hidden, act_fun=act_fun)
        self.combination_block = act.combination_block(mult_size, psi_hidden, act.complex_gelu)
        self.psi = act.ComplexSelfAttention1(psi_hidden, 1, psi_hidden, psi_num_hidden, act.complex_gelu)

    def forward(self, spins, alpha):
        lat_out = self.lattice_net(spins.unsqueeze(1))
        t_out = self.tNN(alpha)
        comb = self.combination_block(lat_out, t_out)
        psi_out = self.psi(comb)
        return (1 - torch.exp(- self.init_decay * alpha[:, :1]) ) * psi_out + self.psi_init(spins, self.lattice_sites)

    def configure_optimizers(self):
        optimizer = self.opt(self.parameters(), lr=self.lr)
        lr_scheduler = self.scheduler(optimizer, step_size=2, verbose=True, gamma=0.5)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'train_loss'}