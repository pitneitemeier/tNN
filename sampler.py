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
        #print('trying to move spins')
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

def suggest_alpha_step(alpha, step_size):
    alpha = alpha + step_size * torch.randn(*alpha.shape, device=alpha.device)
    alpha[alpha > 1] = 2 - alpha[alpha > 1]
    alpha[alpha < 0] = - alpha[alpha < 0]
    return alpha

def single_flip_train(current_sample):
    new_sample = current_sample.clone()
    num_samples = new_sample.shape[0]
    lattice_sites = new_sample.shape[2]
    flip_ind = torch.randint(0, lattice_sites, 
        (num_samples,), device=new_sample.device) + lattice_sites * torch.arange(0, num_samples, device=new_sample.device)
    (new_sample.flatten())[flip_ind] *= -1
    return new_sample

def scale_alphas(alphas, alpha_max, alpha_min):
    return alphas * (alpha_max - alpha_min) + alpha_min

def update_with_accepted(accept, old_tensor, new_tensor):
    return accept * new_tensor + (1 - accept) * old_tensor

class MCTrainSampler(BaseSampler):
    def __init__(self, lattice_sites, batch_size, alpha_step, alpha_max, alpha_min) -> None:
        super().__init__(lattice_sites)
        self.batch_size = batch_size
        self.alpha_step = alpha_step
        self.alpha_max = torch.tensor(alpha_max)
        self.alpha_min = torch.tensor(alpha_min)
        self.alphas = torch.rand((batch_size, 1, self.alpha_max.shape[0]), requires_grad=True)
        self.spins =  2 * torch.randint(0, 2, (batch_size, 1, self.lattice_sites), dtype=torch.get_default_dtype()) - 1
        self.psi = None

    def to(self, device):
        self.device = device
        self.alphas = self.alphas.to(device)
        self.spins = self.spins.to(device)
        if self.psi is not None:
            self.psi = self.psi.to(device)
        self.alpha_max = self.alpha_max.to(device)
        self.alpha_min = self.alpha_min.to(device)

    def _update_samples(self, model):
        if self.psi is None:
            self.psi = model.call_forward(self.spins, self.alphas)
            return
        # generate new samples and calculate psi
        new_alphas = suggest_alpha_step(self.alphas, self.alpha_step)
        new_spins = single_flip_train(self.spins)
        new_psi = model.call_forward(new_spins, new_alphas)

        # decide wether to accept new samples according to metropolis algorithm
        transition_prob = torch.clamp(utils.abs_sq(new_psi) / (utils.abs_sq(self.psi) + 1e-8), 0, 1)
        accept = torch.bernoulli(transition_prob)
        
        # update samples
        self.spins = update_with_accepted(accept, self.spins, new_spins)
        self.alphas = update_with_accepted(accept, self.alphas, new_alphas)
        self.psi = update_with_accepted(accept, self.psi, new_psi)

    def __call__(self, model):
        if self.device != model.device:
            self.to(model.device)
        self._update_samples(model)

        # scale the alphas to the training range
        alphas = scale_alphas(self.alphas, self.alpha_max, self.alpha_min)
        return {'psi': self.psi, 'alphas': alphas, 'spins':self.spins}


        



if __name__ == '__main__':
    pass