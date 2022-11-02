import torch
from abc import ABC, abstractclassmethod

from torch._C import dtype
import utils

class BaseSampler(ABC):
    def __init__(self, lattice_sites) -> None:
        super().__init__()
        self.lattice_sites = lattice_sites
        self.device = 'cpu'
        self.is_MC = False

    @abstractclassmethod
    def __call__(self, model, alpha, val_set_index):
        pass


class ExactSampler(BaseSampler):
    def __init__(self, lattice_sites) -> None:
        super().__init__(lattice_sites)
        self.spins = None

    def to(self, device):
        print('trying to move spins')
        self.spins = self.spins.to(device)

    def __call__(self, model, alpha, val_set_index=0):
        if self.spins is None:
            print("generating spins")
            self.spins = utils.get_all_spin_configs(self.lattice_sites).type(torch.get_default_dtype())
        if not (model.device == self.device):
            self.to(model.device)
        return self.spins.unsqueeze(0)

class RandomSampler(BaseSampler):
    def __init__(self, lattice_sites, num_samples):
        super().__init__(lattice_sites)
        self.num_samples = num_samples
    
    def __call__(self, model, alpha, val_set_index=0):
        return (torch.randint(0, 2, (alpha.shape[0], self.num_samples, self.lattice_sites), 
            device=model.device, dtype=torch.get_default_dtype()) * 2 - 1)


def single_flip(current_sample):
    new_sample = current_sample.clone()
    num_samples = new_sample.shape[0] * new_sample.shape[1]
    flip_ind = torch.randint(0, new_sample.shape[2], 
        (num_samples,), device=new_sample.device) + new_sample.shape[2] * torch.arange(0, num_samples, device=new_sample.device)
    (new_sample.flatten())[flip_ind] *= -1
    return new_sample

class MCMCSampler(BaseSampler):
    '''MCMC Sampler that uses one chain per produced sample'''
    def __init__(self, lattice_sites, num_samples, steps_to_equilibrium):
        super().__init__(lattice_sites)
        self.num_samples = num_samples
        self.steps_to_equilibrium = steps_to_equilibrium
        self.is_MC = True

    def _update_sample(self, model, alpha, current_sample, current_prob):
        proposed_sample = single_flip(current_sample)
        update_prob = utils.abs_sq(model.call_forward(proposed_sample, alpha))
        transition_prob = torch.clamp(update_prob / current_prob, 0, 1)
        accept = torch.bernoulli(transition_prob)
        new_sample = accept * proposed_sample + (1 - accept) * current_sample
        new_prob = accept * update_prob + (1 - accept) * current_prob
        return new_sample, new_prob
        
    def __call__(self, model, alpha, val_set_idx=0):
        sample = (torch.randint(0, 2, (alpha.shape[0], self.num_samples, self.lattice_sites), 
            device=model.device, dtype=torch.get_default_dtype()) * 2 - 1)
        update_prob = utils.abs_sq(model.call_forward(sample, alpha))
        for i in range(self.steps_to_equilibrium):
            sample, update_prob = self._update_sample(model, alpha, sample, update_prob)
        return sample

class MCMCSamplerChains(BaseSampler):
    '''MCMC Sampler that uses one chain per alpha value'''
    def __init__(self, lattice_sites, num_samples, steps_to_equilibrium):
        super().__init__(lattice_sites)
        self.is_MC = True
        self.num_samples = num_samples
        self.steps_to_equilibrium = steps_to_equilibrium

    def _update_sample(self, model, alpha, current_sample, current_prob):
        current_prob = utils.abs_sq(model.call_forward(current_sample, alpha))
        proposed_sample = single_flip(current_sample)
        update_prob = utils.abs_sq(model.call_forward(proposed_sample, alpha))
        transition_prob = torch.clamp(update_prob / current_prob, 0, 1)
        accept = torch.bernoulli(transition_prob)
        new_sample = accept * proposed_sample + (1 - accept) * current_sample
        new_prob = accept * update_prob + (1 - accept) * current_prob
        return new_sample, new_prob
        
    def __call__(self, model, alpha, val_set_index):
        sample = torch.zeros((alpha.shape[0], self.num_samples, self.lattice_sites), 
            device=model.device, dtype=torch.get_default_dtype())
        chains = (torch.randint(0, 2, (alpha.shape[0], 1, self.lattice_sites), 
            device=model.device, dtype=torch.get_default_dtype()) * 2 - 1)
        update_prob = utils.abs_sq(model.call_forward(chains, alpha))
        # thermalization
        for i in range(self.steps_to_equilibrium):
            chains, update_prob = self._update_sample(model, alpha, chains, update_prob)
        #sampling
        for i in range(self.num_samples):
            sample[:, i, :] = chains[:,0,:]
            for i in range(5 * self.lattice_sites):
                chains, update_prob = self._update_sample(model, alpha, chains, update_prob)
        return sample

if __name__ == '__main__':
    pass