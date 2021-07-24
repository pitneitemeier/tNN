import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
import torch
import utils

class Condition():
    def __init__(self):
        pass

    def __call__(self, model, psi_s, spins, alpha):
        print('forward not yet implemented')
        raise NotImplementedError()

    def __str__(self):
        print('unset condition')
        raise NotImplementedError()

    def to(self, device):
        'to not implemented for condition'
        raise NotImplementedError()

class schrodinger_eq(Condition):
    def __init__(self, h_list, lattice_sites, name):
        super().__init__()
        h_tot = sum(h_list, [])
        self.h_mat = utils.get_total_mat_els(h_tot, lattice_sites)
        self.h_map = utils.get_map(h_tot, lattice_sites)
        self.name = name
        self.lattice_sites = lattice_sites

    def to(self, device):
        self.h_map.to(device)
        self.h_mat.to(device)
    
    def __str__(self):
        return f"{self.name} hamiltonian for {self.lattice_sites} lattice sites \n"

    def get_h_mult_from_alpha(self, alpha):
        print('nyi')
        h_mult = 0
        return h_mult

    def __call__(self, model, psi_s, spins, alpha, loss_weight):
        sp_h = utils.get_sp(self.spins, self.h_map)
        psi_sp_h = model.call_forward_sp(sp_h, alpha)
        h_mult = self.get_ext_param_from_alpha(alpha)
        h_loc = utils.calc_Oloc(psi_sp_h, self.h_mat, spins, h_mult)
        dt_psi_s = utils.calc_dt_psi(psi_s, alpha)
        h_loc_sq_sum = (torch.abs(h_loc)**2).sum(1)
        dt_psi_sq_sum = (torch.abs(dt_psi_s)**2).sum(1)
        dt_psi_h_loc_sum = (torch.conj(dt_psi_s) * h_loc).sum(1)
        schroedinger = torch.mean( torch.exp(- loss_weight * alpha[:, 0, 0]) * torch.abs( h_loc_sq_sum + dt_psi_sq_sum - 2 * torch.imag(dt_psi_h_loc_sum) ) ** 2)
        #schroedinger = torch.mean( torch.abs( h_loc_sq_sum + dt_psi_sq_sum - 2 * torch.imag(dt_psi_h_loc_sum) ) ** 2)
        return schroedinger

x = Condition()
#x()
x.to('cuda')
print(x)


class Environment(LightningDataModule):
    def __init__(self, cond_list, h_param_range, t_min=0, t_max=1):
        super().__init__()
        self.cond_list = cond_list
        self.h_param_range = h_param_range
        self.t_min = t_min
        self.t_max = t_max
    
