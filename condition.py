from torch.utils import data
import Datasets
import torch
import utils
import matplotlib.pyplot as plt

from abc import ABC, abstractclassmethod
from matplotlib import cm
import numpy as np
def get_figsize(width, ratio=1.61803):
  return (width, width/ratio)
plt.rcParams.update({'font.size': 8})

class Condition(ABC):
    def __init__(self):
        self.device = 'cpu'

    @abstractclassmethod
    def __call__(self, model, data_dict):
        print('forward not yet implemented')

    @abstractclassmethod
    def __str__(self):
        print('unset condition')
        raise NotImplementedError()
    
    @abstractclassmethod
    def to(self, device):
        'to not implemented for condition'
        raise NotImplementedError()




class schrodinger_mc(Condition):
    def __init__(self, h_list, lattice_sites, sampler, epoch_len, name):
        super().__init__()
        h_tot = sum(h_list, [])
        self.h_mat_list = [utils.get_total_mat_els(h, lattice_sites) for h in h_list]
        self.h_map = utils.get_map(h_tot, lattice_sites)
        self.name = name
        self.epoch_len = epoch_len
        self.sampler = sampler
        self.lattice_sites = lattice_sites

    def to(self, device):
        self.h_map = self.h_map.to(device)
        for i in range(len(self.h_mat_list)):
            self.h_mat_list[i] = self.h_mat_list[i].to(device)
        self.device = device
    
    def __str__(self):
        return f"{self.name} hamiltonian for {self.lattice_sites} lattice sites \n"
    
    def __call__(self, model, _):
        if not (model.device == self.device):
            self.to(model.device)
        data = self.sampler(model)
        schrodinger_residual = utils.schrodinger_residual_mc(model, data['alphas'], data['spins'], data['psi'], data['dt_psi'], self.h_map, self.h_mat_list)
        return torch.mean( utils.abs_sq(schrodinger_residual) )
        

    def get_dataset(self):
        return Datasets.Dummy_Data(epoch_len=self.epoch_len)

class schrodinger_eq_time_dep(Condition):
    def __init__(self, h_list, lattice_sites, sampler, t_range, h_func, epoch_len, name, weight = 1):
        super().__init__()
        self.weight = weight
        h_tot = sum(h_list, [])
        self.h_mat_list = [utils.get_total_mat_els(h, lattice_sites) for h in h_list]
        self.h_map = utils.get_map(h_tot, lattice_sites)
        self.name = name
        self.t_range = t_range
        self.epoch_len = epoch_len
        self.sampler = sampler
        self.lattice_sites = lattice_sites
        self.h_func = h_func

    def to(self, device):
        self.h_map = self.h_map.to(device)
        for i in range(len(self.h_mat_list)):
            self.h_mat_list[i] = self.h_mat_list[i].to(device)
        self.device = device
    
    def __str__(self):
        return f"{self.name} hamiltonian for {self.lattice_sites} lattice sites \n"
    
    def __call__(self, model, data_dict):
        if not (model.device == self.device):
            self.to(model.device)
        t = data_dict['t']
        schrodinger_res_per_config = utils.schrodinger_res_per_config_time_dep(model, t, self.h_func, self.sampler, self.h_map, self.h_mat_list)
        schrodinger_loss = torch.mean( utils.abs_sq(schrodinger_res_per_config) )
        
        return self.weight * schrodinger_loss

    def get_dataset(self):
        return Datasets.Train_Data_time_dep(t_range=self.t_range, epoch_len=self.epoch_len)

class schrodinger_eq_per_config(Condition):
    def __init__(self, h_list, lattice_sites, sampler, t_range, h_param_range, epoch_len, name, weight = 1, exp_decay = False):
        super().__init__()
        self.weight = weight
        h_tot = sum(h_list, [])
        self.h_mat_list = [utils.get_total_mat_els(h, lattice_sites) for h in h_list]
        self.h_map = utils.get_map(h_tot, lattice_sites)
        self.name = name
        self.t_range = t_range
        self.h_param_range = h_param_range
        self.epoch_len = epoch_len
        self.sampler = sampler
        self.lattice_sites = lattice_sites
        self.exp_decay = exp_decay

    def to(self, device):
        self.h_map = self.h_map.to(device)
        for i in range(len(self.h_mat_list)):
            self.h_mat_list[i] = self.h_mat_list[i].to(device)
        self.device = device
    
    def __str__(self):
        return f"{self.name} hamiltonian for {self.lattice_sites} lattice sites \n"
    
    def __call__(self, model, data_dict):
        if not (model.device == self.device):
            self.to(model.device)
        alpha = data_dict['alpha']
        schrodinger_res_per_config = utils.schrodinger_res_per_config(model, alpha, self.sampler, self.h_map, self.h_mat_list)
        if not self.exp_decay:
            schrodinger_loss = torch.mean( utils.abs_sq(schrodinger_res_per_config) )
        else:
            schrodinger_loss = torch.mean( torch.exp( -data_dict['alpha'][:,0]) * utils.abs_sq(schrodinger_res_per_config) )
        
        return self.weight * schrodinger_loss

    def get_dataset(self):
        return Datasets.Train_Data(t_range=self.t_range, h_param_range=self.h_param_range, epoch_len=self.epoch_len)

class ED_Validation_batched(Condition):
    def __init__(self, obs, lattice_sites, ED_data, alpha, val_names, sampler, plot_fmt='.png', name_app='', plot_folder=''):
        super().__init__()
        self.sampler = sampler
        self.MC_sampling = sampler.is_MC
        self.ED_data = torch.from_numpy(ED_data).type(torch.get_default_dtype())
        self.val_names = val_names
        self.alpha = torch.from_numpy(alpha).type(torch.get_default_dtype())
        self.obs_mat = utils.get_total_mat_els(obs, lattice_sites)
        self.obs_map = utils.get_map(obs, lattice_sites)
        self.plot_fmt = plot_fmt
        self.name_app = name_app
        self.plot_folder = plot_folder

    def to(self, device):
        #self.ED_data = self.ED_data.to(device)
        self.obs_map = self.obs_map.to(device)
        self.obs_mat = self.obs_mat.to(device)
        self.device = device

    def __str__(self):
        return f'Condition to match ED_data'
    
    def __call__(self, model, data_dict):
        if not (model.device == self.device):
            self.to(model.device)

        res = []
        loss = 0
        for i, alpha in enumerate(data_dict["alpha"]):
            alpha = alpha.unsqueeze(1) #unsqueeze spin dimension to correspond to expected shape
            observable = model.measure_observable_compiled_batched(alpha, self.sampler, self.obs_mat, self.obs_map)
            loss += torch.mean(torch.abs(data_dict['ED_data'][i] - observable)**2)
            res.append(torch.stack((alpha.squeeze()[:,0],observable,data_dict['ED_data'][i]), dim=1))
        res = torch.stack(res, dim=0)
        loss /= i
        torch.cuda.empty_cache()
        return loss, res

    def get_num_val_sets(self):
        return len(self.val_names)

    def get_val_set_name(self, val_set_idx):
        return self.val_names[val_set_idx]

    def get_dataset(self):
        return Datasets.Simple_Val_Data(self.alpha, self.ED_data)

    def plot_results(self, model, res):
        res = res.cpu()
        fig, ax = plt.subplots(figsize=get_figsize(6))
        ax.set_xlabel('t')
        ax.set_ylabel(r'$ \langle '+ 'S^x' +r' \rangle$', fontsize=13)
        tot_title = f'Magnetization for {model.lattice_sites} Spins'
        ax.set_title(tot_title)
        dummy_lines = [ax.plot([], [], c='black', ls='--', label='ED'), ax.plot([], [], c='black', label='tNN')]
        num_t_values = self.alpha.shape[1]
        for i in range(res.shape[1]):
            val_set_idx = i
            t_arr = res[:,i,0]
            #sorting by t value
            t_arr, ind = t_arr.sort()
            observable = res[:,i,1][ind]
            ED_observable = res[:,i,2][ind]
            #t_arr, observable, ED_observable = res_dict['val_set_idx'][0], res_dict['t'], res_dict['observable'], res_dict['ED_data']
            label = self.get_val_set_name(val_set_idx)
            ax.plot(t_arr, observable, label=label, c=f'C{val_set_idx}')
            ax.plot(t_arr, ED_observable, c=f'C{val_set_idx}', ls='--')
        ax.legend()
        fig.savefig(self.plot_folder + tot_title + self.name_app + self.plot_fmt)
        plt.close(fig)

class Simple_ED_Validation(Condition):
    def __init__(self, obs, lattice_sites, ED_data, alpha, val_names, sampler, plot_fmt='.png', name_app='', plot_folder=''):
        super().__init__()
        self.sampler = sampler
        self.MC_sampling = sampler.is_MC
        self.ED_data = torch.from_numpy(ED_data).type(torch.get_default_dtype())
        self.val_names = val_names
        self.alpha = torch.from_numpy(alpha).type(torch.get_default_dtype())
        self.obs_mat = utils.get_total_mat_els(obs, lattice_sites)
        self.obs_map = utils.get_map(obs, lattice_sites)
        self.plot_fmt = plot_fmt
        self.name_app = name_app
        self.plot_folder = plot_folder

    def to(self, device):
        #self.ED_data = self.ED_data.to(device)
        self.obs_map = self.obs_map.to(device)
        self.obs_mat = self.obs_mat.to(device)
        self.device = device

    def __str__(self):
        return f'Condition to match ED_data'

    def __call__(self, model, data_dict):
        if not (model.device == self.device):
            self.to(model.device)

        res = []
        loss = 0
        for i, alpha in enumerate(data_dict["alpha"]):
            #print(alpha[:,0])
            alpha = alpha.unsqueeze(1) #unsqueeze spin dimension to correspond to expected shape
            observable = model.measure_observable_compiled(alpha, self.sampler, self.obs_mat, self.obs_map)
            loss += torch.mean(torch.abs(data_dict['ED_data'][i] - observable)**2)
            #print("i= ", i)
            #print("alpha shape",alpha.shape)
            #print("observable shape", observable.shape)
            #print("Ed data shape", data_dict['ED_data'][i].shape)
            res.append(torch.stack((alpha.squeeze()[:,0],observable,data_dict['ED_data'][i]), dim=1))
        res = torch.stack(res, dim=0)
        loss /= i
        return loss, res

    def get_num_val_sets(self):
        return len(self.val_names)

    def get_val_set_name(self, val_set_idx):
        return self.val_names[val_set_idx]

    def get_dataset(self):
        return Datasets.Simple_Val_Data(self.alpha, self.ED_data)

    def plot_results(self, model, res):
        res = res.cpu()
        fig, ax = plt.subplots(figsize=get_figsize(6))
        ax.set_xlabel('t')
        ax.set_ylabel(r'$ \langle '+ 'S^x' +r' \rangle$', fontsize=13)
        tot_title = f'Magnetization for {model.lattice_sites} Spins'
        ax.set_title(tot_title)
        dummy_lines = [ax.plot([], [], c='black', ls='--', label='ED'), ax.plot([], [], c='black', label='tNN')]
        num_t_values = self.alpha.shape[1]
        for i in range(res.shape[1]):
            val_set_idx = i
            t_arr = res[:,i,0]
            #sorting by t value
            t_arr, ind = t_arr.sort()
            observable = res[:,i,1][ind]
            ED_observable = res[:,i,2][ind]
            #t_arr, observable, ED_observable = res_dict['val_set_idx'][0], res_dict['t'], res_dict['observable'], res_dict['ED_data']
            label = self.get_val_set_name(val_set_idx)
            ax.plot(t_arr, observable, label=label, c=f'C{val_set_idx}')
            ax.plot(t_arr, ED_observable, c=f'C{val_set_idx}', ls='--')
        ax.legend()
        fig.savefig(self.plot_folder + tot_title + self.name_app + self.plot_fmt)
        plt.close(fig)

class ED_Validation(Condition):
    def __init__(self, obs, lattice_sites, ED_data, alpha, val_names, sampler, plot_fmt='.png'):
        super().__init__()
        self.sampler = sampler
        self.MC_sampling = sampler.is_MC
        self.ED_data = torch.from_numpy(ED_data).type(torch.get_default_dtype())
        self.val_names = val_names
        self.alpha = torch.from_numpy(alpha).type(torch.get_default_dtype())
        self.obs_mat = utils.get_total_mat_els(obs, lattice_sites)
        self.obs_map = utils.get_map(obs, lattice_sites)
        self.plot_fmt = plot_fmt

    def to(self, device):
        #self.ED_data = self.ED_data.to(device)
        self.obs_map = self.obs_map.to(device)
        self.obs_mat = self.obs_mat.to(device)
        self.device = device

    def __str__(self):
        return f'Condition to match ED_data'

    def __call__(self, model, data_dict):
        if not (model.device == self.device):
            self.to(model.device)
        observable = model.measure_observable_compiled(data_dict['alpha'], self.sampler, self.obs_mat, self.obs_map)
        loss = torch.mean(torch.abs(data_dict['ED_data'] - observable)**2)
        data_dict['t'] = data_dict['alpha'][:, 0, 0]
        del data_dict['alpha']
        data_dict['observable'] = observable
        return loss, data_dict

    def get_num_val_sets(self):
        return len(self.val_names)

    def get_val_set_name(self, val_set_idx):
        return self.val_names[val_set_idx]

    def get_dataset(self):
        return Datasets.Val_Data(self.alpha, self.ED_data)

    def plot_results(self, model, res_dict):
        # transform from dict of tensors into list of dicts for sorting
        res_list = []
        for i in range(len(res_dict['val_set_idx'])):
            temp = {}
            for key in res_dict:
                temp[key] = res_dict[key][i]
            res_list.append(temp)

        res_list.sort(key=lambda entry: entry['t'])
        res_list.sort(key=lambda entry: entry['val_set_idx'])
        
        fig, ax = plt.subplots(figsize=get_figsize(6))
        ax.set_xlabel('t')
        ax.set_ylabel(r'$ \langle '+ 'S^x' +r' \rangle$', fontsize=13)
        tot_title = f'Magnetization for {model.lattice_sites} Spins'
        ax.set_title(tot_title)
        dummy_lines = [ax.plot([], [], c='black', ls='--', label='ED'), ax.plot([], [], c='black', label='tNN')]
        num_t_values = self.alpha.shape[1]
        for i in range(int(len(res_list)/num_t_values)):
            res_dict = {key: [dict[key].item() for dict in res_list[i*num_t_values:(i+1)*num_t_values]] for key in res_list[0]}
            #print(res_dict['val_set_idx'])
            val_set_idx, t_arr, observable, ED_observable = res_dict['val_set_idx'][0], res_dict['t'], res_dict['observable'], res_dict['ED_data']
            label = self.get_val_set_name(val_set_idx)
            ax.plot(t_arr, observable, label=label, c=f'C{val_set_idx}')
            ax.plot(t_arr, ED_observable, c=f'C{val_set_idx}', ls='--')
        ax.legend()
        fig.savefig(tot_title + self.plot_fmt)
        plt.close(fig)

class ED_Test(Condition):
    def __init__(self, magn_op, corr_op_list, h_list, lattice_sites, ED_magn, ED_susc, ED_corr, alpha, val_names, sampler, name, plot_folder='', magn_op_name=None, corr_op_name=None, plot_fmt='.png'):
        super().__init__()
        self.sampler = sampler
        self.MC_sampling = sampler.is_MC
        if magn_op_name is None:
            self.magn_name = magn_op[0][0].name
        else:
            self.magn_name = magn_op_name
        if corr_op_name is None:
            self.corr_name = corr_op_list[0][0][0].name
        else:
            self.corr_name = corr_op_name
        self.name = name
        self.plot_folder = plot_folder
        self.ED_magn = torch.from_numpy(ED_magn).type(torch.get_default_dtype())
        self.ED_susc = torch.from_numpy(ED_susc).type(torch.get_default_dtype())
        self.ED_corr = torch.from_numpy(ED_corr).type(torch.get_default_dtype())
        self.val_names = val_names
        self.alpha = torch.from_numpy(alpha).type(torch.get_default_dtype())
        self.magn_mat = utils.get_total_mat_els(magn_op, lattice_sites)
        self.magn_map = utils.get_map(magn_op, lattice_sites)
        self.corr_mat_list = [utils.get_total_mat_els(corr_op, lattice_sites) for corr_op in corr_op_list]
        self.corr_map_list = [utils.get_map(corr_op, lattice_sites) for corr_op in corr_op_list]
        self.h_mat_list = [utils.get_total_mat_els(h, lattice_sites) for h in h_list]
        h_tot = sum(h_list, [])
        self.h_map = utils.get_map(h_tot, lattice_sites)
        self.plot_fmt = plot_fmt


    def to(self, device):
        self.magn_map = self.magn_map.to(device)
        self.magn_mat = self.magn_mat.to(device)
        for i in range(len(self.corr_mat_list)):
            self.corr_mat_list[i] = self.corr_mat_list[i].to(device)
            self.corr_map_list[i] = self.corr_map_list[i].to(device)
        self.h_map = self.h_map.to(device)
        for i in range(len(self.h_mat_list)):
            self.h_mat_list[i] = self.h_mat_list[i].to(device)
        self.device = device

    def __str__(self):
        return f'Condition to match ED_data'

    def __call__(self, model, data_dict):
        if not (model.device == self.device):
            self.to(model.device)
        alpha = data_dict['alpha'].clone()
        alpha.requires_grad = True
        magn = model.measure_observable_compiled(alpha, self.sampler, self.magn_mat, self.magn_map)
        susc = utils.get_susceptibility(magn, alpha)
        corr = [model.measure_observable_compiled(data_dict['alpha'], self.sampler, corr_mat, corr_map) 
            for corr_mat, corr_map in zip(self.corr_mat_list, self.corr_map_list)]
        corr = torch.stack(corr, dim=-1)
        loss = torch.mean(torch.abs(data_dict['ED_magn'] - magn)**2)
        alpha = data_dict['alpha'].clone()
        h_res = torch.max( torch.abs( utils.schrodinger_res_per_config(model, alpha, self.sampler, self.h_map, self.h_mat_list) ), dim=1)[0].squeeze(1)
        energy = model.measure_observable_compiled(alpha, self.sampler, self.h_mat_list, self.h_map)
        assert(self.sampler.is_MC==False)
        spins = self.sampler(model, data_dict['alpha'].clone())
        psi_s = model.call_forward(spins, data_dict['alpha'].clone())
        norm = torch.sum(utils.abs_sq(psi_s), dim=1)
        data_dict['energy'] = energy
        data_dict['norm'] = norm
        #print(h_res.shape)
        data_dict['t'] = data_dict['alpha'][:, 0, 0]
        del data_dict['alpha']
        data_dict['h_res'] = h_res
        data_dict['magn'] = magn
        data_dict['susc'] = susc
        data_dict['corr'] = corr

        return loss, data_dict

    def get_num_val_sets(self):
        return len(self.val_names)

    def get_val_set_name(self, val_set_idx):
        return self.val_names[val_set_idx]
        #return f'{self.val_names[val_set_idx]:.1f}'

    def get_dataset(self):
        return Datasets.Test_Data(self.alpha, self.ED_magn, self.ED_susc, self.ED_corr)


    def plot_results(self, model, res_dict):
        # transform from dict of tensors into list of dicts for sorting
        res_list = []
        for i in range(len(res_dict['val_set_idx'])):
            temp = {}
            for key in res_dict:
                temp[key] = res_dict[key][i]
            res_list.append(temp)

        res_list.sort(key=lambda entry: entry['t'])
        res_list.sort(key=lambda entry: entry['val_set_idx'])
        '''
        fig, ax = plt.subplots(figsize=get_figsize(6))
        ax.set_xlabel('t')
        ax.set_ylabel(r'$ \langle '+ 'S^x' +r' \rangle$')
        tot_title = f'Test magnetization for {model.lattice_sites} Spins'
        ax.set_title(tot_title)
        dummy_lines = [ax.plot([], [], c='black', ls='--', label='ED'), ax.plot([], [], c='black', label='tNN')]
        '''
        num_t_values = self.alpha.shape[1]
        tot_fig, tot_ax = plt.subplots(figsize=get_figsize(6))
        tot_ax.set_xlabel('t')
        
        magn_name = self.magn_name
        corr_name = self.corr_name
        name= self.name

        tot_ax.set_ylabel(r'$ \langle '+ magn_name +r' \rangle$', fontsize=8)
        tot_title = f'{name}, {model.lattice_sites} Spins'
        tot_ax.set_title(tot_title)
        dummy_lines = [tot_ax.plot([], [], c='black', ls='--', label='ED'), tot_ax.plot([], [], c='black', label='tNN')]

        for i in range(int(len(res_list)/num_t_values)):
            #transforming list of dicts back into dict of lists
            res_dict = {key: torch.stack([dict[key] for dict in res_list[i*num_t_values:(i+1)*num_t_values]]).cpu() for key in res_list[0]}
            #print(res_dict['val_set_idx'])
            val_set_idx, t_arr, magn, ED_magn, susc, ED_susc, corr, ED_corr, h_res, energy, norm= \
                res_dict['val_set_idx'][0], res_dict['t'], res_dict['magn'], res_dict['ED_magn'], \
                res_dict['susc'], res_dict['ED_susc'], res_dict['corr'], res_dict['ED_corr'], res_dict['h_res'], \
                res_dict['energy'], res_dict['norm']
            #h_res_int = torch.mean(torch.abs(utils.primitive_fn(t_arr, h_res)), dim=1)
            label = 'h = ' + str(self.get_val_set_name(val_set_idx))
            tot_ax.plot(t_arr, magn, label=label, c=f'C{val_set_idx}', alpha=0.9)
            tot_ax.plot(t_arr, ED_magn, c=f'C{val_set_idx}', ls='--')
            tot_ax.plot(t_arr, ED_magn, c=f'black', ls='--', alpha=0.1)
            fig, ax = plt.subplots(4, 1, figsize=get_figsize(6), sharex='col', gridspec_kw={'height_ratios': [2,2,1,1]})
            fig.subplots_adjust(right=0.85)
            #ax[0,1].axis('off')
            #ax[1,1].axis('off')
            title = tot_title + ', ' + label
            ax[0].set_title(title)

            ax[0].plot(t_arr, magn.detach().cpu(), label='tNN', c='C0')
            ax[0].plot(t_arr, ED_magn, label = 'ED', ls='--', c='C1')
            ax2_1 = ax[0].twinx()
            ax[0].set_zorder(ax2_1.get_zorder() + 1)
            ax[0].patch.set_visible(False)
            ax2_1.tick_params(axis='y', labelcolor='grey')
            ax2_1.set_ylabel('Residuals', c='grey')
            ax2_1.plot(t_arr, ED_magn - magn, label='ED $-$ tNN', c='grey', ls='dotted')
            max_diff = torch.max(torch.abs(ED_magn - magn))
            max_diff *= 1.1
            ax2_1.set_ylim(-max_diff, max_diff)
            ax2_1.axhline(c='grey', alpha=0.3)
            lines, labels = ax[0].get_legend_handles_labels()
            lines2, labels2 = ax2_1.get_legend_handles_labels()
            ax[0].legend(lines + lines2, labels + labels2)
            #ax2_1.plot(t_arr, h_res, label='tNN res', c='grey', ls='dashed')
            #ax2_1.plot(t_arr, h_res_int, label='integrated pred error', c='grey', ls='dashed')
            ax[1].plot(t_arr, susc.detach().cpu(), label='tNN', c='C0')
            ax[1].plot(t_arr, ED_susc, label = 'ED', ls='--', c='C1')
            ax2_1 = ax[1].twinx()
            ax[1].set_zorder(ax2_1.get_zorder() + 1)
            ax[1].patch.set_visible(False)
            ax2_1.tick_params(axis='y', labelcolor='grey')
            ax2_1.set_ylabel('Residuals', c='grey')
            ax2_1.plot(t_arr, ED_susc - susc.detach().cpu().numpy(), label='ED $-$ tNN', c='grey', ls='dotted')
            max_diff = torch.max(torch.abs(ED_susc - susc))
            max_diff *= 1.1
            ax2_1.set_ylim(-max_diff, max_diff)
            ax2_1.axhline(c='grey', alpha=0.3)
            lines, labels = ax[1].get_legend_handles_labels()
            lines2, labels2 = ax2_1.get_legend_handles_labels()
            ax[1].legend(lines + lines2, labels + labels2)
            
            y = np.arange(0,model.lattice_sites/2 + 1) + 0.5
            X, Y = np.meshgrid(t_arr, y)
            c1 = ax[2].pcolor(X, Y, corr.cpu().transpose(0,1)[:, :-1], label='tNN', rasterized=True)
            c2 = ax[3].pcolor(X, Y, ED_corr.cpu().transpose(0,1)[:, :-1], label='ED', rasterized=True)
            cbar_ax = fig.add_axes([0.87, 0.11, 0.02, 0.255])
            fig.colorbar(c2, cax=cbar_ax)
            
            ax[2].set_yticks(y[:-1] + 0.5)
            ax[2].tick_params(axis='y', labelsize=6)
            ax[2].legend()
            ax[3].set_yticks(y[:-1] + 0.5)
            ax[3].tick_params(axis='y', labelsize=6)
            ax[3].legend()
            ax[3].set_xlabel('t')
            ax[0].set_ylabel(r'$ \langle ' + magn_name + r' \rangle$', fontsize=8)
            ax[1].set_ylabel(r'$ \frac{\partial \langle ' + magn_name + r' \rangle}{\partial h}$', fontsize=10)
            ax[2].set_ylabel(r'd')
            ax[3].set_ylabel(r'd')
            
            comm_ax = fig.add_subplot(313, frame_on=False)
            comm_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            comm_ax.set_ylabel(r'$\langle ' + corr_name + '_i \cdot ' + corr_name + r'_{i+d} \rangle $', labelpad=25, fontsize=8)
            fig.savefig(self.plot_folder + title + self.plot_fmt)
            plt.close(fig)
            fig, ax = plt.subplots(2,1, sharex=True)
            ax[0].plot(t_arr, energy/energy[0]-1, label='relative energy')
            ax[1].plot(t_arr, norm/norm[0]-1, label='relative norm')
            ax[1].set_xlabel('t')
            ax[0].set_ylabel('E')
            ax[1].set_ylabel(r'$ \langle \psi | \psi \rangle$')
            fig.savefig(self.plot_folder + title + '_E_norm' +self.plot_fmt)
            fig, ax = plt.subplots()
            ax.plot(torch.abs(ED_magn - magn), h_res)
            fig.savefig(self.plot_folder + title + 'res_h_res' + self.plot_fmt)

        tot_ax.legend()
        tot_fig.tight_layout()
        tot_fig.savefig(self.plot_folder + tot_title + self.plot_fmt)
        plt.close(fig)


class Time_dep_Test(Condition):
    def __init__(self, magn_op, corr_op_list, h_list, lattice_sites, ED_magn, ED_corr, alpha, 
        val_names, sampler, name, plot_folder='', magn_op_name=None, corr_op_name=None, 
        plot_fmt='.png'):
        super().__init__()
        self.sampler = sampler
        self.MC_sampling = sampler.is_MC
        if magn_op_name is None:
            self.magn_name = magn_op[0][0].name
        else:
            self.magn_name = magn_op_name
        if corr_op_name is None:
            self.corr_name = corr_op_list[0][0][0].name
        else:
            self.corr_name = corr_op_name
        self.name = name
        self.plot_folder = plot_folder
        self.ED_magn = torch.from_numpy(ED_magn).type(torch.get_default_dtype())
        self.ED_corr = torch.from_numpy(ED_corr).type(torch.get_default_dtype())
        self.val_names = val_names
        self.alpha = torch.from_numpy(alpha).type(torch.get_default_dtype())
        self.magn_mat = utils.get_total_mat_els(magn_op, lattice_sites)
        self.magn_map = utils.get_map(magn_op, lattice_sites)
        self.corr_mat_list = [utils.get_total_mat_els(corr_op, lattice_sites) for corr_op in corr_op_list]
        self.corr_map_list = [utils.get_map(corr_op, lattice_sites) for corr_op in corr_op_list]
        self.h_mat_list = [utils.get_total_mat_els(h, lattice_sites) for h in h_list]
        h_tot = sum(h_list, [])
        self.h_map = utils.get_map(h_tot, lattice_sites)
        self.plot_fmt = plot_fmt


    def to(self, device):
        self.magn_map = self.magn_map.to(device)
        self.magn_mat = self.magn_mat.to(device)
        for i in range(len(self.corr_mat_list)):
            self.corr_mat_list[i] = self.corr_mat_list[i].to(device)
            self.corr_map_list[i] = self.corr_map_list[i].to(device)
        self.h_map = self.h_map.to(device)
        for i in range(len(self.h_mat_list)):
            self.h_mat_list[i] = self.h_mat_list[i].to(device)
        self.device = device

    def __str__(self):
        return f'Condition to match ED_data'

    def __call__(self, model, data_dict):
        if not (model.device == self.device):
            self.to(model.device)
        alpha = data_dict['alpha'].clone()
        alpha.requires_grad = True
        magn = model.measure_observable_compiled(alpha, self.sampler, self.magn_mat, self.magn_map)
        corr = [model.measure_observable_compiled(data_dict['alpha'], self.sampler, corr_mat, corr_map) 
            for corr_mat, corr_map in zip(self.corr_mat_list, self.corr_map_list)]
        corr = torch.stack(corr, dim=-1)
        loss = torch.mean(torch.abs(data_dict['ED_magn'] - magn)**2)
        alpha = data_dict['alpha'].clone()
        h_res = torch.max( torch.abs( utils.schrodinger_res_per_config(model, alpha, self.sampler, self.h_map, self.h_mat_list) ), dim=1)[0].squeeze(1)
        energy = model.measure_observable_compiled(alpha, self.sampler, self.h_mat_list, self.h_map)
        assert(self.sampler.is_MC==False)
        spins = self.sampler(model, data_dict['alpha'].clone())
        psi_s = model.call_forward(spins, data_dict['alpha'].clone())
        norm = torch.sum(utils.abs_sq(psi_s), dim=1)
        data_dict['energy'] = energy
        data_dict['norm'] = norm
        #print(h_res.shape)
        data_dict['t'] = data_dict['alpha'][:, 0, 0]
        data_dict['h_res'] = h_res
        data_dict['magn'] = magn
        data_dict['corr'] = corr

        return loss, data_dict

    def get_num_val_sets(self):
        return len(self.val_names)

    def get_val_set_name(self, val_set_idx):
        return self.val_names[val_set_idx]
        #return f'{self.val_names[val_set_idx]:.1f}'

    def get_dataset(self):
        return Datasets.Test_Data_simple(self.alpha, self.ED_magn, self.ED_corr)


    def plot_results(self, model, res_dict):
        # transform from dict of tensors into list of dicts for sorting
        res_list = []
        for i in range(len(res_dict['val_set_idx'])):
            temp = {}
            for key in res_dict:
                temp[key] = res_dict[key][i]
            res_list.append(temp)

        res_list.sort(key=lambda entry: entry['t'])
        res_list.sort(key=lambda entry: entry['val_set_idx'])
        '''
        fig, ax = plt.subplots(figsize=get_figsize(6))
        ax.set_xlabel('t')
        ax.set_ylabel(r'$ \langle '+ 'S^x' +r' \rangle$')
        tot_title = f'Test magnetization for {model.lattice_sites} Spins'
        ax.set_title(tot_title)
        dummy_lines = [ax.plot([], [], c='black', ls='--', label='ED'), ax.plot([], [], c='black', label='tNN')]
        '''
        num_t_values = self.alpha.shape[1]
        tot_fig, tot_ax = plt.subplots(figsize=get_figsize(6))
        tot_ax.set_xlabel('t')
        
        magn_name = self.magn_name
        corr_name = self.corr_name
        name= self.name

        tot_ax.set_ylabel(r'$ \langle '+ magn_name +r' \rangle$', fontsize=8)
        tot_title = f'{name}, {model.lattice_sites} Spins'
        tot_ax.set_title(tot_title)
        dummy_lines = [tot_ax.plot([], [], c='black', ls='--', label='ED'), tot_ax.plot([], [], c='black', label='tNN')]

        print(len(res_list),num_t_values)
        for i in range(int(len(res_list)/num_t_values)):
            #transforming list of dicts back into dict of lists
            res_dict = {key: torch.stack([dict[key] for dict in res_list[i*num_t_values:(i+1)*num_t_values]]).cpu() for key in res_list[0]}
            #print(res_dict['val_set_idx'])
            val_set_idx, t_arr, alpha, magn, ED_magn, corr, ED_corr, h_res, energy, norm= \
                res_dict['val_set_idx'][0], res_dict['t'], res_dict['alpha'], res_dict['magn'], res_dict['ED_magn'], \
                res_dict['corr'], res_dict['ED_corr'], res_dict['h_res'], \
                res_dict['energy'], res_dict['norm']
            #h_res_int = torch.mean(torch.abs(utils.primitive_fn(t_arr, h_res)), dim=1)
            label = 'h = ' + str(self.get_val_set_name(val_set_idx))
            tot_ax.plot(t_arr, magn, label=label, c=f'C{val_set_idx}', alpha=0.9)
            tot_ax.plot(t_arr, ED_magn, c=f'C{val_set_idx}', ls='--')
            tot_ax.plot(t_arr, ED_magn, c=f'black', ls='--', alpha=0.1)
            fig, ax = plt.subplots(4, 1, figsize=get_figsize(6), sharex='col', gridspec_kw={'height_ratios': [2,2,1,1]})
            fig.subplots_adjust(right=0.85)
            #ax[0,1].axis('off')
            #ax[1,1].axis('off')
            title = tot_title
            ax[0].set_title(title)
            title = tot_title + ', ' + label

            ax[0].plot(t_arr, magn.detach().cpu(), label='tNN', c='C0')
            ax[0].plot(t_arr, ED_magn, label = 'ED', ls='--', c='C1')
            ax2_1 = ax[0].twinx()
            ax[0].set_zorder(ax2_1.get_zorder() + 1)
            ax[0].patch.set_visible(False)
            ax2_1.tick_params(axis='y', labelcolor='grey')
            ax2_1.set_ylabel('Residuals', c='grey')
            ax2_1.plot(t_arr, ED_magn - magn, label='ED $-$ tNN', c='grey', ls='dotted')
            max_diff = torch.max(torch.abs(ED_magn - magn))
            max_diff *= 1.1
            ax2_1.set_ylim(-max_diff, max_diff)
            ax2_1.axhline(c='grey', alpha=0.3)
            lines, labels = ax[0].get_legend_handles_labels()
            lines2, labels2 = ax2_1.get_legend_handles_labels()
            ax[0].legend(lines + lines2, labels + labels2)
            ax[1].plot(alpha[:,0,0], alpha[:,0,1])
            ax[1].set_ylabel('$h(t)$')
            
            y = np.arange(0,model.lattice_sites/2 + 1) + 0.5
            X, Y = np.meshgrid(t_arr, y)
            c1 = ax[2].pcolor(X, Y, corr.cpu().transpose(0,1)[:, :-1], label='tNN', rasterized=True)
            c2 = ax[3].pcolor(X, Y, ED_corr.cpu().transpose(0,1)[:, :-1], label='ED', rasterized=True)
            cbar_ax = fig.add_axes([0.87, 0.11, 0.02, 0.255])
            fig.colorbar(c2, cax=cbar_ax)
            
            ax[2].set_yticks(y[:-1] + 0.5)
            ax[2].tick_params(axis='y', labelsize=6)
            ax[2].legend()
            ax[3].set_yticks(y[:-1] + 0.5)
            ax[3].tick_params(axis='y', labelsize=6)
            ax[3].legend()
            ax[3].set_xlabel('t')
            ax[0].set_ylabel(r'$ \langle ' + magn_name + r' \rangle$', fontsize=8)
            ax[2].set_ylabel(r'd')
            ax[3].set_ylabel(r'd')
            
            comm_ax = fig.add_subplot(313, frame_on=False)
            comm_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            comm_ax.set_ylabel(r'$\langle ' + corr_name + '_i \cdot ' + corr_name + r'_{i+d} \rangle $', labelpad=25, fontsize=8)
            fig.savefig(self.plot_folder + title + self.plot_fmt)
            plt.close(fig)
            fig, ax = plt.subplots(2,1, sharex=True)
            ax[0].plot(t_arr, energy/energy[0]-1, label='relative energy')
            ax[1].plot(t_arr, norm/norm[0]-1, label='relative norm')
            ax[1].set_xlabel('t')
            ax[0].set_ylabel('E')
            ax[1].set_ylabel(r'$ \langle \psi | \psi \rangle$')
            fig.savefig(self.plot_folder + title + '_E_norm' +self.plot_fmt)
            fig, ax = plt.subplots()
            ax.plot(torch.abs(ED_magn - magn), h_res)
            fig.savefig(self.plot_folder + title + 'res_h_res' + self.plot_fmt)

        tot_ax.legend()
        tot_fig.tight_layout()
        tot_fig.savefig(self.plot_folder + tot_title + self.plot_fmt)
        plt.close(fig)

class magn_surface(Condition):
    def __init__(self, magn_op, h_range, t_range, lattice_sites, sampler, name, plot_folder=''):
        super().__init__()
        resolution = 100
        self.t_arr = torch.linspace(t_range[0], t_range[1], resolution)
        self.h_arr = torch.linspace(h_range[0], h_range[1], resolution)
        self.alpha = torch.stack((self.t_arr.repeat(resolution,1), self.h_arr.unsqueeze(1).repeat(1,resolution)), dim=2).unsqueeze(-2)
        self.sampler = sampler
        self.MC_sampling = sampler.is_MC
        self.magn_name = magn_op[0][0].name
        self.name = name
        self.plot_folder = plot_folder
        self.magn_mat = utils.get_total_mat_els(magn_op, lattice_sites)
        self.magn_map = utils.get_map(magn_op, lattice_sites)

    def to(self, device):
        self.magn_map = self.magn_map.to(device)
        self.magn_mat = self.magn_mat.to(device)
        self.device = device

    def __str__(self):
        return f'Condition used for a surface plot of magn'

    def __call__(self, model, data_dict):
        if not (model.device == self.device):
            self.to(model.device)
        magn = model.measure_observable_compiled(data_dict['alpha'], self.sampler, self.magn_mat, self.magn_map)
        data_dict['t'] = data_dict['alpha'][:, 0, 0]
        data_dict['h'] = data_dict['alpha'][:, 0, 1]
        del data_dict['alpha']
        data_dict['magn'] = magn

        return 0, data_dict

    def get_dataset(self):
        return Datasets.Simple_Data(self.alpha)


    def plot_results(self, model, res_dict):
        # transform from dict of tensors into list of dicts for sorting
        res_list = []
        for i in range(len(res_dict['val_set_idx'])):
            temp = {}
            for key in res_dict:
                temp[key] = res_dict[key][i]
            res_list.append(temp)

        res_list.sort(key=lambda entry: entry['t'])
        res_list.sort(key=lambda entry: entry['val_set_idx'])

        num_t_values = self.alpha.shape[1]
        Z = np.empty((len(self.t_arr), len(self.h_arr)))
        for i in range(int(len(res_list)/num_t_values)):
            #transforming list of dicts back into dict of lists
            key = 'magn'
            Z[i] = torch.stack([dict[key] for dict in res_list[i*num_t_values:(i+1)*num_t_values]]).cpu()
        fig, ax = plt.subplots(figsize=(7,5), subplot_kw={"projection": "3d"})

        X = self.t_arr
        Y = self.h_arr
        X, Y = np.meshgrid(X, Y)
        ax.set_xlim(0,3)
        ax.set_ylim(0.2,1.2)
        ax.set_zlim(0,1)
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z,
                            linewidth=0, antialiased=True, cmap=cm.inferno, rasterized=True, rcount=100, ccount=100)
        #print(Z)
        #ax.plot(xs=X[0,:], ys=Y[0,:], zs=Z[0,:], c='r')
        ax.set_xlabel('t')
        ax.set_ylabel('h')
        ax.set_zlabel(r'$ \langle ' + self.magn_name + r' \rangle$', fontsize=8)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)

        fig.savefig(self.name+'magn_surface.pdf')

class surface_animation(Condition):
    def __init__(self, magn_op, h_range, t_range, lattice_sites, sampler, name, plot_folder=''):
        super().__init__()
        resolution = 100
        self.t_arr = torch.linspace(t_range[0], t_range[1], resolution)
        self.h_arr = torch.linspace(0.2, 1.2, resolution)
        self.alpha = torch.stack((self.t_arr.repeat(resolution,1), self.h_arr.unsqueeze(1).repeat(1,resolution)), dim=2).unsqueeze(-2)
        self.sampler = sampler
        self.MC_sampling = sampler.is_MC
        self.magn_name = magn_op[0][0].name
        self.name = name
        self.plot_folder = plot_folder
        self.magn_mat = utils.get_total_mat_els(magn_op, lattice_sites)
        self.magn_map = utils.get_map(magn_op, lattice_sites)
        self.batch_counter = 0 

    def to(self, device):
        self.magn_map = self.magn_map.to(device)
        self.magn_mat = self.magn_mat.to(device)
        self.device = device

    def __str__(self):
        return f'Condition used for a surface plot of magn'

    def __call__(self, model, data_dict):
        if not (model.device == self.device):
            self.to(model.device)
        magn = model.measure_observable_compiled(data_dict['alpha'], self.sampler, self.magn_mat, self.magn_map)
        data_dict['t'] = data_dict['alpha'][:, 0, 0]
        data_dict['h'] = data_dict['alpha'][:, 0, 1]
        del data_dict['alpha']
        data_dict['magn'] = magn

        return 0, data_dict

    def get_dataset(self):
        return Datasets.Simple_Data(self.alpha)


    def plot_results(self, model, res_dict):
        self.batch_counter +=1
        # transform from dict of tensors into list of dicts for sorting
        res_list = []
        for i in range(len(res_dict['val_set_idx'])):
            temp = {}
            for key in res_dict:
                temp[key] = res_dict[key][i]
            res_list.append(temp)

        res_list.sort(key=lambda entry: entry['t'])
        res_list.sort(key=lambda entry: entry['val_set_idx'])

        num_t_values = self.alpha.shape[1]
        Z = np.empty((len(self.t_arr), len(self.h_arr)))
        for i in range(int(len(res_list)/num_t_values)):
            #transforming list of dicts back into dict of lists
            key = 'magn'
            Z[i] = torch.stack([dict[key] for dict in res_list[i*num_t_values:(i+1)*num_t_values]]).cpu()
        np.savetxt(self.plot_folder + self.name + str(self.batch_counter) + '.csv', Z, delimiter=',')


#old:
'''
class schrodinger_eq(Condition):
    def __init__(self, h_list, lattice_sites, name, weight = 1):
        super().__init__(weight)
        h_tot = sum(h_list, [])
        self.name = name
        self.lattice_sites = lattice_sites
        self.num_op_h_list = [len(x) for x in h_list]
        self.num_summands_h = int(torch.tensor(self.num_op_h_list).sum())
        self.h_mat = utils.get_total_mat_els(h_tot, lattice_sites)
        self.h_map = utils.get_map(h_tot, lattice_sites)
        self.h_mult = torch.zeros((1,1, self.num_summands_h))

    def to(self, device):
        self.h_map = self.h_map.to(device)
        self.h_mat = self.h_mat.to(device)
        self.h_mult = self.h_mult.to(device)
        self.device = device
    
    def __str__(self):
        return f"{self.name} hamiltonian for {self.lattice_sites} lattice sites \n"

    def update_h_mult(self, alpha):
        self.h_mult = self.h_mult.new_ones(alpha.shape[0], 1, self.num_summands_h)
        current_ind = self.num_op_h_list[0]
        for i in range(alpha.shape[2] - 1):
            self.h_mult[:, :, current_ind : current_ind + self.num_op_h_list[i+1]] = alpha[:, 0, i+1].reshape(-1, 1, 1)
            current_ind += self.num_op_h_list[i+1]
    
    def __call__(self, model, psi_s, spins, alpha):
        if not (model.device == self.device):
            self.to(model.device)
        sp_h = utils.get_sp(spins, self.h_map)
        psi_sp_h = model.call_forward_sp(sp_h, alpha)
        self.update_h_mult(alpha)
        h_loc = utils.calc_Oloc(psi_sp_h, self.h_mat, spins, self.h_mult)
        dt_psi_s = utils.calc_dt_psi(psi_s, alpha)
        h_loc_sq_sum = (utils.abs_sq(h_loc)).sum(1)
        dt_psi_sq_sum = (utils.abs_sq(dt_psi_s)).sum(1)
        dt_psi_h_loc_sum = (torch.conj(dt_psi_s) * h_loc).sum(1)
        schroedinger_loss = torch.mean( torch.abs( h_loc_sq_sum + dt_psi_sq_sum - 2 * torch.imag(dt_psi_h_loc_sum) ) )
        return self.weight * schroedinger_loss


class init_observable(Condition):
    def __init__(self, obs, lattice_sites, name, init_t = 0, init_value = 1, weight=1):
        super().__init__(weight)
        self.name = name
        self.lattice_sites = lattice_sites
        self.init_t = init_t
        self.init_value = init_value
        self.obs_mat = utils.get_total_mat_els(obs, lattice_sites)
        self.obs_map = utils.get_map(obs, lattice_sites)

    def to(self, device):
        self.obs_map = self.obs_map.to(device)
        self.obs_mat = self.obs_mat.to(device)
        self.device = device
    
    def __str__(self):
        return f"{self.name} inital condition for {self.lattice_sites} lattice sites \n"

    def __call__(self, model, psi_s, spins, alpha):
        if not (model.device == self.device):
            self.to(model.device)
        alpha_init = alpha.detach().clone()
        alpha_init[:, :, 0] = self.init_t
        psi_s_init = model.call_forward(spins, alpha_init)
        sp_obs = utils.get_sp(spins, self.obs_map)
        psi_sp_obs = model.call_forward_sp(sp_obs, alpha_init)
        obs_loc = utils.calc_Oloc(psi_sp_obs, self.obs_mat, spins)
        psi_s_init_sq_sum = (utils.abs_sq(psi_s_init)).sum(1)
        psi_init_obs_loc_sum = (torch.conj(psi_s_init) * obs_loc).sum(1)
        init_cond_loss = torch.mean( (utils.abs_sq( (psi_init_obs_loc_sum * (1 / psi_s_init_sq_sum)) - self.init_value)) )
        return self.weight * init_cond_loss

class init_psi_per_config(Condition):
    def __init__(self, psi_init, lattice_sites, name, init_t=0, weight=1):
        super().__init__(weight)
        self.psi_init = psi_init
        self.lattice_sites = lattice_sites
        self.name = name
        self.init_t = init_t

    def to(self, device):
        self.device = device

    def __str__(self):
        return f'{self.name} initial condition per spin config'

    def __call__(self, model, psi_s, spins, alpha):
        if not (model.device == self.device):
            self.to(model.device)
        alpha_init = alpha.detach().clone()
        alpha_init[:, :, 0] = self.init_t
        psi_s_init_target = self.psi_init(spins)
        psi_s_init = model.call_forward(spins, alpha_init)
        init_cond_per_time = utils.abs_sq(psi_s_init - psi_s_init_target)
        init_cond = torch.mean(init_cond_per_time)
        return self.weight * init_cond

class init_scalar_prod(Condition):
    def __init__(self, psi_init, lattice_sites, name, init_t=0, weight=1):
        super().__init__(weight)
        self.psi_init = psi_init
        self.lattice_sites = lattice_sites
        self.name = name
        self.init_t = init_t
        self.psi_s_init_target = None

    def to(self, device):
        self.device = device

    def __str__(self):
        return f'{self.name} initial condition via scalar prod'

    def __call__(self, model, psi_s, spins, alpha):
        if not (model.device == self.device):
            self.to(model.device)
        alpha_init = alpha.detach().clone()
        alpha_init[:, :, 0] = self.init_t
        if self.psi_s_init_target is None:
            self.psi_s_init_target = self.psi_init(spins).unsqueeze(2)
        psi_s_init = model.call_forward(spins, alpha_init)
        psi_s_init_target_sum = ( torch.abs(self.psi_s_init_target)**2 ).sum(1)
        psi_s_init_sum = ( utils.abs_sq(psi_s_init) ).sum(1)
        psi_s_init_psi_s_target_sum = (torch.conj(psi_s_init) * self.psi_s_init_target).sum(1) 
        init_cond = torch.mean( torch.abs( psi_s_init_sum + psi_s_init_target_sum - 2 * torch.real( psi_s_init_psi_s_target_sum ) ) )
        return self.weight * init_cond

class Norm(Condition):
    def __init__(self, norm_target=1, weight=1):
        super().__init__(weight)
        self.norm_target = norm_target
        self.name='Norm'
    
    def to(self, device):
        self.device = device
    
    def __str__(self):
        return f'Norm_target = {self.norm_target}'

    def __call__(self, model, psi_s, spins, alpha):
        if not (model.device == self.device):
            self.to(model.device)
        batched_norm = utils.psi_norm_sq(psi_s)
        norm_loss = torch.mean( (batched_norm - self.norm_target**2) ** 2 )
        return self.weight * norm_loss
'''