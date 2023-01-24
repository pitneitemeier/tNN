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

class ExactBatchedSampler(BaseSampler):
    def __init__(self, lattice_sites, batch_size) -> None:
        assert(2**lattice_sites % batch_size == 0)
        super().__init__(lattice_sites)
        self.batch_size = batch_size
        self.num_batches = 2**lattice_sites // batch_size

    def __call__(self, model, alphas, batch):
        mask = 2**torch.arange(self.lattice_sites, device=model.device)
        spins = torch.arange(batch * self.batch_size, (batch + 1)* self.batch_size, 1, device=model.device)
        spins = 2*(spins.unsqueeze(-1).bitwise_and(mask).ne(0)) - 1
        return spins.unsqueeze(0).to(dtype=torch.get_default_dtype())



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
        
    def __call__(self, model, alpha):
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
    #alpha = alpha + step_size * (torch.rand(*alpha.shape, device=alpha.device) - .5)
    alpha[alpha > 1] = 2 - alpha[alpha > 1]
    alpha[alpha < 0] = - alpha[alpha < 0]
    return alpha

def suggest_new_alpha(alpha, _):
    return torch.rand_like(alpha)

def single_flip_train(sample):
    num_samples = sample.shape[0]
    lattice_sites = sample.shape[2]
    flip_ind = torch.randint(0, lattice_sites, 
        (num_samples,), device=sample.device) + lattice_sites * torch.arange(0, num_samples, device=sample.device)
    (sample.flatten())[flip_ind] = (sample.flatten())[flip_ind] * (-1)
    return sample

def scale_alphas(alphas, alpha_max, alpha_min):
    return alphas * (alpha_max - alpha_min) + alpha_min

def update_with_accepted(accept, old_tensor, new_tensor):
    return torch.where(accept, new_tensor.detach(), old_tensor.detach())

class MCTrainSampler(BaseSampler):
    def __init__(self, lattice_sites, batch_size, alpha_step, alpha_max, alpha_min) -> None:
        super().__init__(lattice_sites)
        self.batch_size = batch_size
        self.alpha_step = alpha_step
        self.alpha_max = torch.tensor(alpha_max)
        self.alpha_min = torch.tensor(alpha_min)
        self.alphas = scale_alphas(torch.rand((batch_size, 1, self.alpha_max.shape[0]), requires_grad=True), self.alpha_max, self.alpha_min)
        self.spins =  2 * torch.randint(0, 2, (batch_size, 1, self.lattice_sites), dtype=torch.get_default_dtype()) - 1
        self.psi = None
        self.dt_psi = None

    def to(self, device):
        self.device = device
        self.alphas = self.alphas.to(device)
        self.spins = self.spins.to(device)
        if self.psi is not None:
            self.psi = self.psi.to(device)
        self.alpha_max = self.alpha_max.to(device)
        self.alpha_min = self.alpha_min.to(device)

    def _update_samples(self, model):
        # generate new samples and calculate psi
        #new_alphas = suggest_alpha_step(self.alphas.detach(), self.alpha_step)
        new_alphas = scale_alphas(suggest_new_alpha(self.alphas.detach(), self.alpha_step), self.alpha_max, self.alpha_min)
        new_alphas.requires_grad = True
        new_alphas.grad = None
        new_spins = single_flip_train(self.spins.detach())
        new_psi = model.call_forward(new_spins, new_alphas)
        new_dt_psi = utils.calc_dt_psi(new_psi, new_alphas).detach()
        
        self.alphas = self.alphas.detach()
        self.alphas.requires_grad = True
        self.alphas.grad = None
        old_psi = model.call_forward(self.spins, self.alphas)
        old_dt_psi = utils.calc_dt_psi(old_psi, self.alphas).detach()

        # decide wether to accept new samples according to metropolis algorithm
        transition_prob = torch.clamp(utils.abs_sq(new_psi.detach()) / (utils.abs_sq(old_psi.detach()) + 1e-8), 0, 1)
        accept = torch.bernoulli(transition_prob).type(torch.bool)
        accept_ratio = accept.sum()/accept.numel()
        model.log('accept_ratio', accept_ratio, prog_bar=True)
        
        # update samples
        self.spins = update_with_accepted(accept, self.spins.detach(), new_spins.detach())
        self.alphas = update_with_accepted(accept, self.alphas.detach(), new_alphas.detach())
        self.psi = update_with_accepted(accept, old_psi.detach(), new_psi.detach())
        self.dt_psi = update_with_accepted(accept, old_dt_psi.detach(), new_dt_psi.detach())

    def __call__(self, model):
        if self.device != model.device:
            self.to(model.device)
        self._update_samples(model)

        return {'alphas': self.alphas.detach(), 'spins':self.spins.detach(), 'dt_psi':self.dt_psi.detach(), 'psi':self.psi.detach()}
    
class MCTrainSamplerChains(BaseSampler):
    def __init__(self, lattice_sites, samples_per_chain, chains, alpha_step, alpha_max, alpha_min) -> None:
        super().__init__(lattice_sites)
        self.samples_per_chain = samples_per_chain
        self.chains = chains
        self.alpha_step = alpha_step
        self.alpha_max = torch.tensor(alpha_max)
        self.alpha_min = torch.tensor(alpha_min)
        self.alphas = scale_alphas(torch.rand((self.chains, 1, self.alpha_max.shape[0]), requires_grad=True), self.alpha_max, self.alpha_min)
        self.spins =  2 * torch.randint(0, 2, (self.chains, 1, self.lattice_sites), dtype=torch.get_default_dtype()) - 1
        self.psi = None
        self.dt_psi = None

    def to(self, device):
        self.device = device
        self.alphas = self.alphas.to(device)
        self.spins = self.spins.to(device)
        if self.psi is not None:
            self.psi = self.psi.to(device)
        self.alpha_max = self.alpha_max.to(device)
        self.alpha_min = self.alpha_min.to(device)

    def _update_samples(self, model):
        # generate new samples and calculate psi
        #new_alphas = suggest_alpha_step(self.alphas.detach(), self.alpha_step)
        new_alphas = scale_alphas(suggest_new_alpha(self.alphas.detach(), self.alpha_step), self.alpha_max, self.alpha_min)
        new_alphas.requires_grad = True
        new_alphas.grad = None
        new_spins = single_flip_train(self.spins.detach())
        new_psi = model.call_forward(new_spins, new_alphas)
        new_dt_psi = utils.calc_dt_psi(new_psi, new_alphas).detach()
        
        self.alphas = self.alphas.detach()
        self.alphas.requires_grad = True
        self.alphas.grad = None
        old_psi = model.call_forward(self.spins, self.alphas)
        old_dt_psi = utils.calc_dt_psi(old_psi, self.alphas).detach()

        # decide wether to accept new samples according to metropolis algorithm
        transition_prob = torch.clamp(utils.abs_sq(new_psi.detach()) / (utils.abs_sq(old_psi.detach()) + 1e-8), 0, 1)
        accept = torch.bernoulli(transition_prob).type(torch.bool)
        accept_ratio = accept.sum()/accept.numel()
        model.log('accept_ratio', accept_ratio, prog_bar=True)
        
        # update samples
        self.spins = update_with_accepted(accept, self.spins.detach(), new_spins.detach())
        self.alphas = update_with_accepted(accept, self.alphas.detach(), new_alphas.detach())
        self.psi = update_with_accepted(accept, old_psi.detach(), new_psi.detach())
        self.dt_psi = update_with_accepted(accept, old_dt_psi.detach(), new_dt_psi.detach())

    def __call__(self, model):
        if self.device != model.device:
            self.to(model.device)
        alphas = []
        spins = []
        for _ in range(self.samples_per_chain):
            self._update_samples(model)
            alphas.append(self.alphas.detach())
            spins.append(self.spins.detach())
        #print(f"{alphas[0].shape=}, {len(alphas)=}")
        #print(f"{spins[0].shape=}, {len(spins)=}")
        return {'alphas': torch.cat(alphas, dim=0), 'spins': torch.cat(spins, dim=0), 'dt_psi':None, 'psi':None}   

class ExactTrainSampler(BaseSampler):
    def __init__(self, lattice_sites, batch_size, alpha_max, alpha_min) -> None:
        super().__init__(lattice_sites)
        self.batch_size = batch_size
        self.alpha_max = torch.tensor(alpha_max)
        self.alpha_min = torch.tensor(alpha_min)
        self.alphas = torch.linspace(0,1,batch_size).reshape(-1,1,1).repeat(1,1,self.alpha_max.shape[0])
        self.spins = utils.get_all_spin_configs(lattice_sites).unsqueeze(1).type(torch.get_default_dtype())

    def to(self, device):
        self.device = device
        self.alphas = self.alphas.to(device)
        self.spins = self.spins.to(device)
        self.alpha_max = self.alpha_max.to(device)
        self.alpha_min = self.alpha_min.to(device)

    def __call__(self, model):
        if self.device != model.device:
            self.to(model.device)

        # scale the alphas to the training range
        alphas = scale_alphas(self.alphas, self.alpha_max, self.alpha_min)
        alphas = alphas.unsqueeze(1).repeat(1, 2 ** self.lattice_sites, 1, 1).flatten(0,1)
        spins = self.spins.unsqueeze(0).repeat(self.batch_size, 1, 1, 1).flatten(0,1)
        return {'alphas': alphas.detach(), 'spins':spins.detach()}
        

class RandomTrainSampler(BaseSampler):
    def __init__(self, lattice_sites, batch_size, alpha_max, alpha_min) -> None:
        super().__init__(lattice_sites)
        self.batch_size = batch_size
        self.alpha_max = torch.tensor(alpha_max)
        self.alpha_min = torch.tensor(alpha_min)

    def to(self, device):
        self.device = device
        self.alpha_max = self.alpha_max.to(device)
        self.alpha_min = self.alpha_min.to(device)

    def __call__(self, model):
        if self.device != model.device:
            self.to(model.device)

        # scale the alphas to the training range
        alphas = torch.rand((self.batch_size, 1, self.alpha_max.shape[0]), device=model.device)
        alphas = scale_alphas(alphas, self.alpha_max, self.alpha_min)
        spins = 2 * torch.randint(0, 2, (self.batch_size, 1, self.lattice_sites), dtype=torch.get_default_dtype(), device=model.device) - 1
        return {'alphas': alphas.detach(), 'spins':spins.detach()}

if __name__ == '__main__':
    pass