from typing import Optional
from numpy.lib.shape_base import expand_dims
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader
import torch
import utils
import Datasets
import matplotlib.pyplot as plt
from collections import Sequence
import sampler
import collections
import gc

class Environment(LightningDataModule):
    def __init__(self, train_condition, val_condition, batch_size, val_batch_size, test_batch_size=None, num_workers=0, test_condition=None):
        super().__init__()
        self.train_condition = train_condition
        self.val_condition = val_condition
        self.test_condition = test_condition
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.val_batch_size = val_batch_size

    def setup(self, stage: Optional[str] = None):
        self.train_data = self.train_condition.get_dataset()
        self.val_data = self.val_condition.get_dataset()
        if self.test_condition is not None:
            self.test_data = self.test_condition.get_dataset()

    def train_dataloader(self):
        return DataLoader(self.train_data, self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, self.val_batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.test_condition is not None:
            return DataLoader(self.test_data, self.test_batch_size, num_workers=self.num_workers)
        else:
            return None

class Wave_Fun(LightningModule):
    def __init__(self, lattice_sites, name):
        super().__init__()
        self.lattice_sites = lattice_sites
        self.name = name

    def call_forward(self, spins, alpha):
        '''
        makes forward callable with two batch dimensions (num_alpha_configs, num_spin_configs)
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
        makes forward callable with three batch dimensions for the calculations of psi_s' (num_alpha_configs, num_spin_configs, num_sprimes)
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
        torch.set_grad_enabled(True)
        #loss, res = self.trainer.datamodule.val_condition(self, data_dict)
        loss, _ = self.trainer.datamodule.val_condition(self, data_dict)
        self.log('val_loss', loss, prog_bar=True)
        return loss
        #return res
    
    '''
    def validation_epoch_end(self, res):
        res = torch.cat(res, dim=0)
        res = self.all_gather(res)
        if len(res.shape)==4:
            #print("gathering tensors from all devices")
            res = res.flatten(0,1)
        
        if self.global_rank==0:
            self.trainer.datamodule.val_condition.plot_results(self, res)
    '''
            
    def test_step(self, data_dict, index):
        torch.set_grad_enabled(True)
        loss, res = self.trainer.datamodule.test_condition(self, data_dict)
        self.log('test_loss', loss, prog_bar=True)
        return res
    
    def test_epoch_end(self, res):
        res = torch.cat(res, dim=0)
        res = self.all_gather(res)
        if len(res.shape)==4:
            #print("gathering tensors from all devices")
            res = res.flatten(0,1)
        
        if self.global_rank==0:
            self.trainer.datamodule.test_condition.plot_results(self, res)

    def measure_observable_compiled(self, alpha, sampler, obs_mat, obs_map):
        '''
        allows the measurement of observables of type Operator that have been compiled to mat_els and a map using utils.get_total_mat_els and utils.get_map
        Parameters
        ----------
        alpha: tensor, dtype=float
            inputs to the wave function (time, ext_param)
            shape = (num_alpha_configs, 1, num_inputs)
        sampler : samplers.BaseSampler 
            sampler to get spin configs for alpha values. Can be exact (all spin configs) or any type of MC sampler. 
            Calculation of expectation values is automatically adjusted according to sampler.is_MC
        returns
        -------
        observable : tensor
            the observable
            shape = (num_alpha_configs)
        '''
        ext_param_scale = None
        if isinstance(obs_mat, list):
            #assume if it is list then it has to be scaled with external params of alpha
            ext_param_scale = utils.calc_h_mult(self, alpha, obs_mat)
            obs_mat = torch.cat(obs_mat, dim=3)
        spins = sampler(self, alpha)
        sp_o = utils.get_sp(spins, obs_map)
        psi_sp_o = self.call_forward_sp(sp_o, alpha)
        psi_s = self.call_forward(spins, alpha)
        if not sampler.is_MC:
            o_loc = utils.calc_Oloc(psi_sp_o, obs_mat, spins, ext_param_scale=ext_param_scale)
            psi_sq_sum = (torch.abs(psi_s) ** 2).sum(1)
            psi_s_o_loc_sum = (torch.conj(psi_s) * o_loc).sum(1)
            observable = ( psi_s_o_loc_sum * (1 / psi_sq_sum) ).squeeze(1)
        else:
            o_loc = utils.calc_Oloc_MC(psi_sp_o, psi_s, obs_mat, spins, ext_param_scale=ext_param_scale)
            observable = o_loc.mean(1).squeeze()
        return torch.real(observable) 

    def measure_observable_compiled_batched(self, alpha, sampler, obs_mat, obs_map):
        '''
        allows the measurement of observables of type Operator that have been compiled to mat_els and a map using utils.get_total_mat_els and utils.get_map
        Parameters
        ----------
        alpha: tensor, dtype=float
            inputs to the wave function (time, ext_param)
            shape = (num_alpha_configs, 1, num_inputs)
        sampler : samplers.BaseSampler 
            sampler to get spin configs for alpha values. Can be exact (all spin configs) or any type of MC sampler. 
            Calculation of expectation values is automatically adjusted according to sampler.is_MC
        returns
        -------
        observable : tensor
            the observable
            shape = (num_alpha_configs)
        '''
        ext_param_scale = None
        if isinstance(obs_mat, list):
            #assume if it is list then it has to be scaled with external params of alpha
            ext_param_scale = utils.calc_h_mult(self, alpha, obs_mat)
            obs_mat = torch.cat(obs_mat, dim=3)
        if not sampler.is_MC:
            psi_sq_sum = torch.zeros((alpha.shape[0], 1), device=self.device, dtype=torch.complex128)
            psi_s_o_loc_sum = torch.zeros((alpha.shape[0], 1), device=self.device, dtype=torch.complex128)
            for i in range(sampler.num_batches):
                spins = sampler(self, alpha, i)
                sp_o = utils.get_sp(spins, obs_map)
                psi_sp_o = self.call_forward_sp(sp_o, alpha)
                o_loc = utils.calc_Oloc(psi_sp_o, obs_mat, spins, ext_param_scale=ext_param_scale)
                psi_s = self.call_forward(spins, alpha)
                psi_sq_sum += (torch.abs(psi_s) ** 2).sum(1)
                psi_s_o_loc_sum += (torch.conj(psi_s) * o_loc).sum(1)
            observable = ( psi_s_o_loc_sum * (1 / psi_sq_sum) ).squeeze(1)
        else:
            print('nyi!')
        return torch.real(observable) 
    





