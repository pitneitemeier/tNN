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
    def __init__(self, condition_list, h_param_range, val_condition, val_t_arr, val_h_params, batch_size, train_sampler, val_sampler, t_range = (0,1), num_workers=0, epoch_len=100000):
        super().__init__()
        self.condition_list = condition_list
        self.val_condition = val_condition
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.h_param_range = h_param_range
        self.t_range = t_range
        self.batch_size = batch_size
        self.val_h_params = val_h_params
        self.val_t_arr = val_t_arr
        self.num_workers = num_workers
        self.epoch_len = epoch_len

    def setup(self, stage: Optional[str] = None):
        self.train_data = Datasets.Train_Data(self.t_range, self.h_param_range, self.epoch_len)
        self.val_data = Datasets.Val_Data(self.val_h_params)

    def train_dataloader(self):
        return DataLoader(self.train_data, self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, 1, num_workers=self.num_workers)

"""
def wave_function(Model):
    class wave_fun(Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            #self.spins = utils.get_all_spin_configs(self.lattice_sites).unsqueeze(0).type(torch.get_default_dtype())
            self.register_buffer('spins', utils.get_all_spin_configs(self.lattice_sites).unsqueeze(0).type(torch.get_default_dtype()))
            self.save_hyperparameters()
    
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

        def training_step(self, alpha, batch_idx):
            t_range = self.trainer.datamodule.t_range
            loss_weight = 0
            alpha[:, :, 0] = (t_range[1] - t_range[0]) * alpha[:, :, 0] + t_range[0]
            #broadcast alpha to spin shape. cannot work on view as with spins since it requires grad
            if (alpha.shape[1] == 1):
                alpha = alpha.repeat(1, self.spins.shape[1], 1)
            #gradient needed for dt_psi
            alpha.requires_grad = True

            #calculate psi_s only once since it is needed for many conditions
            psi_s = self.call_forward(self.spins, alpha)
            
            loss = torch.zeros(1, device=self.device, requires_grad=True)
            for condition in self.trainer.datamodule.condition_list:
                cond_loss = condition(self, psi_s, self.spins, alpha, loss_weight)
                self.log(condition.name, cond_loss, logger=True, prog_bar=True)
                loss = loss + cond_loss
            
            self.log('train_loss', loss, logger=True)
            #self.log('end_time', t_end, logger=True, prog_bar=True)
            return {'loss' :loss}

        def validation_step(self, alpha, val_set_idx):
            alpha = alpha[0]
            loss = torch.zeros(1, device=self.device)
            for val_condition in self.trainer.datamodule.val_condition_list:
                loss = loss + val_condition(self, self.spins, alpha, val_set_idx)
            self.log('val_loss', loss, logger=True, prog_bar=True)
            return {'val_loss': loss}
        
        def on_validation_epoch_end(self):
            for condition in self.trainer.datamodule.val_condition_list:
                condition.plot_results()
            
        def measure_observable(self, alpha, obs, lattice_sites):

            alpha = alpha.to(self.device)
            obs_map = utils.get_map(obs, lattice_sites)
            obs_mat = utils.get_total_mat_els(obs, lattice_sites)
            obs_map = obs_map.to(self.device)
            obs_mat = obs_mat.to(self.device)
            sp_o = utils.get_sp(self.spins, obs_map)
            psi_s = self.call_forward(self.spins, alpha)
            psi_sp_o = self.call_forward_sp(sp_o, alpha)
            o_loc = utils.calc_Oloc(psi_sp_o, obs_mat, self.spins)
            psi_sq_sum = (torch.abs(psi_s) ** 2).sum(1)
            psi_s_o_loc_sum = (torch.conj(psi_s) * o_loc).sum(1)
            observable = ( psi_s_o_loc_sum * (1 / psi_sq_sum) ).squeeze(1)
            return torch.real(observable)
        '''
        def to(self, device):
            print(f'moving model to {device}')
            if self.trainer is not None:
                for condition in self.trainer.datamodule.condition_list:
                    condition.to(device)
                for val_condition in self.trainer.datamodule.val_condition_list:
                    val_condition.to(device)
            #self.spins = self.spins.to(device)
            return super().to(device)
        '''
  
    return wave_fun
"""

class Wave_Fun(LightningModule):
    def __init__(self, lattice_sites):
        super().__init__()
        self.lattice_sites = lattice_sites
        #self.register_buffer('spins', utils.get_all_spin_configs(self.lattice_sites).unsqueeze(0).type(torch.get_default_dtype()))

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

    def training_step(self, alpha, batch_idx):
        spins = self.trainer.datamodule.train_sampler(self, alpha, batch_idx)
        #broadcast alpha to spin shape. cannot work on view as with spins since it requires grad
        if (alpha.shape[1] == 1):
            alpha = alpha.repeat(1, spins.shape[1], 1)
        #gradient needed for dt_psi
        alpha.requires_grad = True

        #calculate psi_s only once since it is needed for many conditions
        psi_s = self.call_forward(spins, alpha)
        
        loss = torch.zeros(1, device=self.device, requires_grad=True)
        for condition in self.trainer.datamodule.condition_list:
            cond_loss = condition(self, psi_s, spins, alpha)
            self.log(condition.name, cond_loss, logger=True, prog_bar=True)
            loss = loss + cond_loss
        
        self.log('train_loss', loss, logger=True)
        return {'loss' :loss}

    def validation_step(self, _, val_set_idx):
        alpha = self.trainer.datamodule.val_condition.get_alpha(self, val_set_idx)
        spins = self.trainer.datamodule.val_sampler(self, alpha, val_set_idx)

        val_dict = self.trainer.datamodule.val_condition(self, spins, val_set_idx)
        self.log('val_loss', val_dict['val_loss'], prog_bar=True)
        return val_dict
    
    def validation_epoch_end(self, outs):
        #outs = self.all_gather(outs)
        fig, ax = plt.subplots(figsize=(10,6))
        ax.set_xlabel('ht')
        ax.set_ylabel(r'$ \langle '+ 'S^x' +r' \rangle$', fontsize=13)
        tot_title = f'Magnetization for {self.lattice_sites} Spins'
        ax.set_title(tot_title)
        dummy_lines = [ax.plot([], [], c='black', ls='--', label='ED'), ax.plot([], [], c='black', label='tNN')]
        i = 0
        for out in outs:
            h_param, t_arr, observable, ED_observable = out['val_h_param'], out['t_arr'], out['observable'], out['ED_observable']
            if isinstance(h_param, Sequence):
                label = f'$h={utils.tensor_to_string(h_param)} h_c$'
            else:
                label = f'$h={h_param:.1f} h_c$'
            ax.plot(t_arr.cpu(), observable.cpu(), label=label, c=f'C{i}')
            ax.plot(t_arr.cpu(), ED_observable.cpu(), c=f'C{i}', ls='--')
            i+=1

        ax.legend()
        fig.savefig(tot_title + f'_{self.local_rank}.png')
        plt.close(fig)
        '''
        fig, ax = plt.subplots()
        alpha = self.trainer.datamodule.val_condition.get_alpha(self, 0)
        spins = torch.ones((1,1, self.lattice_sites), device=self.device)
        prob = utils.abs_sq( self.call_forward(spins, alpha))
        print(alpha.shape, prob.shape)
        ax.plot(alpha[:, 0].cpu(), prob[:,0].cpu())
        fig.savefig('prob.png')
        
        plt.close(fig)
        '''
        
    def measure_observable(self, alpha, spins, obs, lattice_sites):
        alpha = alpha.to(self.device)
        spins = spins.to(self.device)
        obs_map = utils.get_map(obs, lattice_sites)
        obs_mat = utils.get_total_mat_els(obs, lattice_sites)
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


