import Datasets
import torch
import utils
import matplotlib.pyplot as plt
from abc import ABC, abstractclassmethod

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


        
class schrodinger_eq_per_config(Condition):
    def __init__(self, h_list, lattice_sites, sampler, t_range, h_param_range, epoch_len, name, weight = 1):
        super().__init__()
        self.weight = weight
        h_tot = sum(h_list, [])
        self.name = name
        self.t_range = t_range
        self.h_param_range = h_param_range
        self.epoch_len = epoch_len
        self.sampler = sampler
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
    
    def __call__(self, model, data_dict):
        if not (model.device == self.device):
            self.to(model.device)
        alpha = data_dict['alpha']
        spins = self.sampler(self, alpha)

        #broadcast alpha to spin shape. cannot work on view as with spins since it requires grad
        if (alpha.shape[1] == 1):
            alpha = alpha.repeat(1, spins.shape[1], 1)
        #gradient needed for dt_psi
        alpha.requires_grad = True

        psi_s = model.call_forward(spins, alpha)
        sp_h = utils.get_sp(spins, self.h_map)
        psi_sp_h = model.call_forward_sp(sp_h, alpha)
        self.update_h_mult(alpha)
        h_loc = utils.calc_Oloc(psi_sp_h, self.h_mat, spins, self.h_mult)
        dt_psi_s = utils.calc_dt_psi(psi_s, alpha)
        schroedinger_per_config = utils.abs_sq(dt_psi_s + 1j * h_loc)
        schroedinger_loss = torch.mean(schroedinger_per_config)
        return self.weight * schroedinger_loss

    def get_dataset(self):
        return Datasets.Train_Data(self.t_range, self.h_param_range, self.epoch_len)



class ED_Validation(Condition):
    def __init__(self, obs, lattice_sites, ED_data, alpha, val_names, sampler):
        super().__init__()
        self.sampler = sampler
        self.MC_sampling = sampler.is_MC
        self.ED_data = torch.from_numpy(ED_data).type(torch.get_default_dtype())
        self.val_names = val_names
        self.alpha = torch.from_numpy(alpha).type(torch.get_default_dtype())
        self.obs_mat = utils.get_total_mat_els(obs, lattice_sites)
        self.obs_map = utils.get_map(obs, lattice_sites)

    def to(self, device):
        self.ED_data = self.ED_data.to(device)
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
        return f'{self.val_names[val_set_idx]:.1f}'

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
        
        fig, ax = plt.subplots(figsize=(10,6))
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
        fig.savefig(tot_title + f'.png')
        plt.close(fig)

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