import torch
from abc import ABC, abstractclassmethod

from torch._C import dtype
import utils

class BaseSampler(ABC):
    def __init__(self, lattice_sites) -> None:
        super().__init__()
        self.lattice_sites = lattice_sites
        self.device = 'cpu'

    @abstractclassmethod
    def __call__(self, model, alpha):
        pass


class ExactSampler(BaseSampler):
    def __init__(self, lattice_sites) -> None:
        super().__init__(lattice_sites)
        self.spins = None

    def to(self, device):
        self.spins = self.spins.to(device)

    def __call__(self, model, alpha):
        if self.spins is None:
           self.spins = utils.get_all_spin_configs(self.lattice_sites).type(torch.get_default_dtype())
        if not (model.device == self.device):
            self.to(model.device)
        return self.spins.unsqueeze(0)

class RandomSampler(BaseSampler):
    def __init__(self, lattice_sites, num_samples):
        super().__init__(lattice_sites)
        self.num_samples = num_samples
    
    def __call__(self, model, alpha):
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
    def __init__(self, lattice_sites, num_samples, steps_to_equilibrium):
        super().__init__(lattice_sites)
        self.num_samples = num_samples
        self.steps_to_equilibrium = steps_to_equilibrium

    def _update_sample(self, model, alpha, current_sample, current_prob):
        print_until = 45
        print_from = 40
        #print(alpha.shape)
        #print('current sample: ', current_sample[print_from:print_until,0,:], current_sample.shape)
        proposed_sample = single_flip(current_sample)
        #proposed_sample = (torch.randint(0, 2, (alpha.shape[0], self.num_samples, self.lattice_sites), 
        #    device=model.device, dtype=torch.get_default_dtype()) * 2 - 1)
        #print('proposed sample: ', proposed_sample[print_from:print_until,0,:])
        update_prob = utils.abs_sq(model.call_forward(proposed_sample, alpha))
        #print('current_prob', current_prob[print_from:print_until,0,:])
        #print('update_prob', update_prob[print_from:print_until,0,:])
        transition_prob = torch.clamp(update_prob / current_prob, 0, 1)
        accept = torch.bernoulli(transition_prob)
        #rand_val = torch.rand_like(update_prob)
        #print('randval: ', rand_val[print_from:print_until,0,:])
        #accept = (update_prob / current_prob > rand_val).type(torch.get_default_dtype())
        #print('accept: ', accept[print_from:print_until,0,:], accept.shape)
        new_sample = accept * proposed_sample + (1 - accept) * current_sample
        #print('new sample', new_sample[print_from:print_until,0,:], new_sample.shape)
        #print('accept*proposed', (accept * proposed_sample)[print_from:print_until,0,:])
        new_prob = accept * update_prob + (1 - accept) * current_prob
        #print('new_prob', new_prob[print_from:print_until,0,:])
        return new_sample, new_prob
        
    def __call__(self, model, alpha):
        #print(alpha)
        #print(alpha.shape)
        sample = (torch.randint(0, 2, (alpha.shape[0], self.num_samples, self.lattice_sites), 
            device=model.device, dtype=torch.get_default_dtype()) * 2 - 1)
        update_prob = utils.abs_sq(model.call_forward(sample, alpha))
        #print('sample_shape', sample.shape)
        #print('prob_shape', update_prob.shape)
        for i in range(self.steps_to_equilibrium):
            sample, update_prob = self._update_sample(model, alpha, sample, update_prob)
        return sample

class MCMCSamplerChains(BaseSampler):
    def __init__(self, lattice_sites, num_samples, steps_to_equilibrium):
        super().__init__(lattice_sites)
        self.num_samples = num_samples
        self.steps_to_equilibrium = steps_to_equilibrium

    def _update_sample(self, model, alpha, current_sample, current_prob):
        print_until = 45
        print_from = 40
        #print(alpha.shape)
        #print('current sample: ', current_sample[print_from:print_until,0,:], current_sample.shape)
        proposed_sample = single_flip(current_sample)
        #proposed_sample = (torch.randint(0, 2, (alpha.shape[0], self.num_samples, self.lattice_sites), 
        #    device=model.device, dtype=torch.get_default_dtype()) * 2 - 1)
        #print('proposed sample: ', proposed_sample[print_from:print_until,0,:])
        update_prob = utils.abs_sq(model.call_forward(proposed_sample, alpha))
        #print('current_prob', current_prob[print_from:print_until,0,:])
        #print('update_prob', update_prob[print_from:print_until,0,:])
        transition_prob = torch.clamp(update_prob / current_prob, 0, 1)
        accept = torch.bernoulli(transition_prob)
        #rand_val = torch.rand_like(update_prob)
        #print('randval: ', rand_val[print_from:print_until,0,:])
        #accept = (update_prob / current_prob > rand_val).type(torch.get_default_dtype())
        #print('accept: ', accept[print_from:print_until,0,:], accept.shape)
        new_sample = accept * proposed_sample + (1 - accept) * current_sample
        #print('new sample', new_sample[print_from:print_until,0,:], new_sample.shape)
        #print('accept*proposed', (accept * proposed_sample)[print_from:print_until,0,:])
        new_prob = accept * update_prob + (1 - accept) * current_prob
        #print('new_prob', new_prob[print_from:print_until,0,:])
        return new_sample, new_prob
        
    def __call__(self, model, alpha):
        #print(alpha.shape)
        sample = torch.zeros((alpha.shape[0], self.num_samples, self.lattice_sites), 
            device=model.device, dtype=torch.get_default_dtype())
        chains = (torch.randint(0, 2, (alpha.shape[0], 1, self.lattice_sites), 
            device=model.device, dtype=torch.get_default_dtype()) * 2 - 1)
        update_prob = utils.abs_sq(model.call_forward(chains, alpha))
        # thermalization
        for i in range(self.steps_to_equilibrium):
            chains, update_prob = self._update_sample(model, alpha, chains, update_prob)
        for i in range(self.num_samples):
            sample[:, i, :] = chains[:,0,:]
            for i in range(5 * self.lattice_sites):
                chains, update_prob = self._update_sample(model, alpha, chains, update_prob)
        return sample

if __name__ == '__main__':
    import models
    
    #rand_sampler = RandomSampler(4, 4)
    model = models.multConvDeep(4, 1, 1e-3)
    model.to('cuda:1')
    alpha = torch.rand((10, 1, 2), device=model.device)
    print(alpha.device, model.device)
    #sample = rand_sampler.__call__(alpha, model)
    #print(sample.shape, sample.device, sample)
    #spins = (torch.randint(0, 2, (10, 1, 4), dtype=torch.get_default_dtype()) * 2 - 1)
    #print(spins)
    #spins = single_flip(spins)
    #print(spins)
    
    sampler = MCMCSampler(4, 2, 20)
    spins = sampler.__call__(model, alpha)
    print(spins)

