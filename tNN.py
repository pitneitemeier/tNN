from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader
import torch
import utils
import Datasets
import matplotlib.pyplot as plt
from collections import Sequence
import sampler

class Environment(LightningDataModule):
    
    def __init__(self, train_condition, val_condition, train_batch_size, val_batch_size, num_workers=0):
        super().__init__()
        self.train_condition = train_condition
        self.val_condition = val_condition
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        self.train_data = self.train_condition.get_dataset()
        self.val_data = self.val_condition.get_dataset()

    def train_dataloader(self):
        return DataLoader(self.train_data, self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, self.val_batch_size, num_workers=self.num_workers)

class Wave_Fun(LightningModule):
    def __init__(self, lattice_sites):
        super().__init__()
        self.lattice_sites = lattice_sites

    def call_forward(self, spins, alpha):
        '''
        makes forward callable with (num_alpha_configs, num_spin_configs)
        Parameters
        ----------
        spins: tensor, dtype=float
            tensor of input spins to wave function 
            shape = (num_alpha_configs / 1, num_spin_configs, lattice_sites)
        alpha: tensor, dtype=float
            other inputs to hamiltonian e.g. (time, ext_param) 
            shape = (num_alpha_configs, num_spin_configs / 1, num_inputs)
        Returns
        -------
        psi: tensor, dtype=complex
            wave function for a combination of (spins, alpha) 
            size = (num_alpha_configs, num_spin_configs, 1)
        '''
        spin_shape = spins.shape
        alpha_shape = alpha.shape
        spins_expanded = spins.expand(alpha_shape[0], spin_shape[1], spin_shape[2])
        alpha_expanded = alpha.expand(alpha_shape[0], spin_shape[1], alpha_shape[2])
        
        spins_flat = torch.flatten(spins_expanded, end_dim=-2)
        alpha_flat = torch.flatten(alpha_expanded, end_dim=-2)
        
        psi = self(spins_flat, alpha_flat)
        return psi.reshape( alpha_shape[0], spin_shape[1], 1)

    def call_forward_sp(self, sprimes, alpha):
        '''
        makes forward callable with (num_alpha_configs, num_spin_configs, num_sprimes)
        Parameters
        ----------
        sprimes: tensor, dtype=float
            tensor of input spins to wave function 
            shape = (num_alpha_configs / 1, num_spin_configs, num_sprimes, lattice_sites)
        alpha: tensor, dtype=float
            other inputs to hamiltonian e.g. (time, ext_param) are broadcasted to s' shape
            shape = (num_alpha_configs, num_spin_configs / 1, num_inputs)

        Returns
        -------
        psi: tensor, dtype=complex
            wave function for a combination of (alpha, spins) 
            size = (num_alpha_configs, num_spin_configs, num_sprimes, 1)
        '''
        sprimes_shape = sprimes.shape
        alpha_shape = alpha.shape
        alpha = alpha.unsqueeze(2)
        alpha_expanded = alpha.expand(alpha_shape[0], sprimes_shape[1], sprimes_shape[2], alpha_shape[2])
        sprimes_expanded = sprimes.expand(alpha_shape[0], sprimes_shape[1], sprimes_shape[2], sprimes_shape[3])

        sprimes_flat = torch.flatten(sprimes_expanded, end_dim=-2)
        alpha_flat = torch.flatten(alpha_expanded, end_dim=-2)
        
        psi = self(sprimes_flat, alpha_flat)
        return psi.reshape( alpha_shape[0], sprimes_shape[1], sprimes_shape[2], 1)

    def training_step(self, data_dict, batch_idx):
        loss = self.trainer.datamodule.train_condition(self, data_dict)
        self.log('train_loss', loss, logger=True)
        return {'loss' :loss}

    def validation_step(self, data_dict, index):
        loss, res_dict = self.trainer.datamodule.val_condition(self, data_dict)
        self.log('val_loss', loss, prog_bar=True)
        return res_dict
    
    def validation_epoch_end(self, outs):
        #TODO make this less ugly
        res_dict = {key: [] for key in outs[0].keys()}
        #put all batches in a list in one dict
        for i, out in enumerate(outs):
            for key in out:
                res_dict[key].append(out[key])
        #concatenate the list of batch tensors into one
        for key in res_dict:
            res_dict[key] = torch.cat(res_dict[key])
        
        res_dict = self.all_gather(res_dict)
        #with multi gpu training need to flatten batch axis from multiple gpus after gathering
        if len(res_dict['val_set_idx'].shape)==2:
            for key in res_dict:
                res_dict[key] = res_dict[key].flatten(0,1)
        
        self.trainer.datamodule.val_condition.plot_results(self, res_dict)

        
    def measure_observable(self, alpha, spins, obs):
        alpha = alpha.to(self.device)
        spins = spins.to(self.device)
        obs_map = utils.get_map(obs, self.lattice_sites)
        obs_mat = utils.get_total_mat_els(obs, self.lattice_sites)
        obs_map = obs_map.to(self.device)
        obs_mat = obs_mat.to(self.device)
        sp_o = utils.get_sp(spins, obs_map)
        psi_s = self.call_forward(spins, alpha)
        psi_sp_o = self.call_forward_sp(sp_o, alpha)
        o_loc = utils.calc_Oloc(psi_sp_o, obs_mat, spins)
        psi_sq_sum = (torch.abs(psi_s) ** 2).sum(1)
        psi_s_o_loc_sum = (torch.conj(psi_s) * o_loc).sum(1)
        observable = ( psi_s_o_loc_sum * (1 / psi_sq_sum) ).squeeze(1)
        return torch.real(observable)

    def measure_observable_compiled(self, alpha, sampler, obs_mat, obs_map):
        spins = sampler(self, alpha)
        sp_o = utils.get_sp(spins, obs_map)
        psi_s = self.call_forward(spins, alpha)
        psi_sp_o = self.call_forward_sp(sp_o, alpha)
        if not sampler.is_MC:
            o_loc = utils.calc_Oloc(psi_sp_o, obs_mat, spins)
            psi_sq_sum = (torch.abs(psi_s) ** 2).sum(1)
            psi_s_o_loc_sum = (torch.conj(psi_s) * o_loc).sum(1)
            observable = ( psi_s_o_loc_sum * (1 / psi_sq_sum) ).squeeze(1)
        else:
            o_loc = utils.calc_Oloc_MC(psi_sp_o, psi_s, obs_mat, spins)
            num_samples = o_loc.shape[1]
            observable = o_loc.sum(1)/num_samples
        return torch.real(observable) 




