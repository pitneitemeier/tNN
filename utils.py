import torch
import numpy as np

def get_map(operator, lattice_sites, device='cpu'):
  '''
  Parameters
  ----------
  operator : Operator.Operator
  lattice_sites : int
  Returns 
  -------
    map : tensor
      map from s -> s' for given operator
      shape = (1, 1, num_summands_o, lattice_sites)
      two dummy dimensions for batch
  '''
  map = torch.ones((1, 1, len(operator), lattice_sites), dtype=torch.int8, device=device)
  for i, op_tuple in enumerate(operator):
    for op in op_tuple:
      map[:, :, i, op.lat_site] = op.switching_map
  return map


def get_total_mat_els(operator, lattice_sites, device='cpu'):
#1.dim: alpha Batch
#2.dim: spin batch
#3.dim: two possibilities for mat el depending on input first=+1, second=-1
#4.dim: Number of summands in operator
#5.dim: number of lattice sites
  mat_els = torch.ones((1, 1, 2, len(operator), lattice_sites), dtype=torch.complex64, device=device)
  for i, op_tuple in enumerate(operator):
    for op in op_tuple:
      mat_els[:, :, :, i, op.lat_site] = op.mat_els
  return mat_els


def get_one_hot(spin_config):
  '''
  Parameters
  ----------
  spin_config : tensor
    shape = (num_alphas, num_spin_configs, lattice_sites)
  Returns
  -------
  spin_one_hot : tensor
    shape = (num_alphas, num_spin_configs, 2, 1, num_lattice sites)
    2 -> two possibilities for mat el depending on input: first=+1, second=-1
    1 dummy dimension for number of summands in hamiltonian (= #s')
  '''
  spin_one_hot = torch.stack((0.5*(spin_config+1), 0.5*(1-spin_config)), dim=2)
  return spin_one_hot.unsqueeze(3)


#getting s_primes for O_loc via map
def get_sp(spin_config, map):
  '''
  Parameters
  ----------
  spin_config : tensor
    shape = (num_alphas, num_spin_configs, lattice_sites)
  map : tensor
    map of the hamiltonian to get s' from each s
    shape = (num_alphas, num_spin_configs, num_summands_h, num_lattice sites)
  Returns
  -------
  sprimes : tensor
    shape = (num_alphas, num_spin_configs, num_sprimes, num_lattice sites)
    with num_sprimes = num_summands_h
  '''
  spin_config = spin_config.unsqueeze(2)
  return map*spin_config


def calc_Oloc(psi_sp, mat_els, spin_config, ext_param_scale = None):
  '''
  Calculates the local Operator value
  Parameters
  ----------
  psi_sp : tensor
    shape = (num_alphas, num_spin_configs, num_sp, 1)
    num_sp = num_summands_o
  mat_els : tensor
    shape = (num_alphas, num_spin_configs, 2, num_sp, num_lattice_sites)
  spin_config : tensor
    shape = (num_alphas, num_spin_configs, lattice_sites)
  ext_param_scale : tensor
    tensor with the external parameters of the hamiltonian to scale up matrix elements
    shape = (num_alphas, num_spin_configs, num_sp, 1)
  Returns
  -------
  O_loc : tensor
    the local energy
    shape = (num_alphas, num_spin_configs, 1)
  '''
  #using onehot encoding of spin config to select the correct mat_el from all mat_els table by multiplication
  s_onehot = get_one_hot(spin_config)
  #summing out zeroed values in dim 2
  res = (mat_els * s_onehot).sum(2)
  #product of all matrix elements for one summand of the hamiltonian.
  res = res.prod(3)

  #scaling all summands of the hamiltonian with external params
  if ext_param_scale is not None:
    res = res * ext_param_scale

  #multiplying with corresponding wave function and summing over s' for each input configuration
  O_loc = (torch.conj(res.unsqueeze(3)) * psi_sp).sum(2)
  return O_loc

def calc_Oloc_MC(psi_sp, psi_s, mat_els, spin_config, ext_param_scale = None):
  '''
  Calculates the local Operator value divided by psi for MCMC Sampling
  Parameters
  ----------
  psi_sp : tensor
    shape = (num_alphas, num_spin_configs, num_sp, 1)
    num_sp = num_summands_o
  mat_els : tensor
    shape = (num_alphas, num_spin_configs, 2, num_sp, num_lattice_sites)
  spin_config : tensor
    shape = (num_alphas, num_spin_configs, lattice_sites)
  ext_param_scale : tensor
    tensor with the external parameters of the hamiltonian to scale up matrix elements
    shape = (num_alphas, num_spin_configs, num_sp, 1)
  Returns
  -------
  O_loc : tensor
    the local energy
    shape = (num_alphas, num_spin_configs, 1)
  '''
  #using onehot encoding of spin config to select the correct mat_el from all mat_els table by multiplication
  s_onehot = get_one_hot(spin_config)
  #summing out zeroed values in dim 2
  res = (mat_els * s_onehot).sum(2)
  #product of all matrix elements for one summand of the hamiltonian.
  res = res.prod(3)

  #scaling all summands of the hamiltonian with external params
  if ext_param_scale is not None:
    res = res * ext_param_scale

  #multiplying with corresponding wave function and summing over s' for each input configuration
  O_loc = (torch.conj(res.unsqueeze(3)) * psi_sp).sum(2)/psi_s
  return O_loc

"""
from itertools import permutations
#get all basis configurations to calculate stochastic mean
def get_all_spin_configs(num_lattice_sites):
  perm = np.zeros( (np.power( 2, num_lattice_sites ), num_lattice_sites) )
  currently_filled = 0
  for i in range(num_lattice_sites + 1):
    tmp = np.ones(num_lattice_sites)
    tmp[i:] = -1
    tmp = np.array(list(set(permutations(tmp))))
    for row in tmp:
      perm[currently_filled, :] = row
      currently_filled += 1
  return torch.from_numpy(perm)
"""

def get_all_spin_configs(lattice_sites):
    mask = 2**torch.arange(lattice_sites)
    return 2*(torch.arange(0,2**lattice_sites).unsqueeze(-1).bitwise_and(mask).ne(0)) - 1


def calc_dt_psi(psi_s, alpha):
  #need to do real and imaginary part separately so pytorch can treat it as a real differentiation
  dt_psi_s_real = torch.autograd.grad(torch.real(psi_s).sum(), alpha, create_graph=True)[0][:,:, 0]
  dt_psi_s_imag = torch.autograd.grad(torch.imag(psi_s).sum(), alpha, create_graph=True)[0][:,:, 0]
  dt_psi_s = dt_psi_s_real + 1.j * dt_psi_s_imag
  return dt_psi_s.unsqueeze(2)

def calc_dt_psi_time_dep(psi_s, t):
  #need to do real and imaginary part separately so pytorch can treat it as a real differentiation
  dt_psi_s_real = torch.autograd.grad(torch.real(psi_s).sum(), t, create_graph=True)[0]
  dt_psi_s_imag = torch.autograd.grad(torch.imag(psi_s).sum(), t, create_graph=True)[0]
  dt_psi_s = dt_psi_s_real + 1.j * dt_psi_s_imag
  return dt_psi_s.unsqueeze(2)



def tensor_to_string(alist):
    format_list = ['{:.1f}' for item in alist] 
    s = ', '.join(format_list)
    return s.format(*alist)

def get_susceptibility(magnetization, alpha):
  return torch.autograd.grad(magnetization.sum(), alpha, retain_graph=True)[0][:, :, -1].sum(1)

def calc_h_mult(model, alpha, h_mat_list):
    assert(alpha.shape[1]==1)
    assert(alpha.shape[2]==len(h_mat_list))
    num_summands = sum([tensor.shape[3] for tensor in h_mat_list])
    h_mult = torch.ones(alpha.shape[0], 1, num_summands, device=model.device)
    current_ind = h_mat_list[0].shape[3]
    for i in range(alpha.shape[2] - 1):
        next_ind = h_mat_list[i+1].shape[3]
        h_mult[:, :, current_ind : current_ind + next_ind] = alpha[:, :, i+1:i+2]
        current_ind += next_ind
    return h_mult

def schrodinger_residual_mc(model, alphas, spins, psi_s, dt_psi_s, h_map, h_mat_list):
    sp_h = get_sp(spins, h_map)
    psi_sp_h = model.call_forward_sp(sp_h, alphas)
    
    h_mult = calc_h_mult(model, alphas, h_mat_list)
    h_mat = torch.cat(h_mat_list, dim=3)
    h_loc = calc_Oloc(psi_sp_h, h_mat, spins, h_mult)
    
    alphas.requires_grad = True
    psi_s = model.call_forward(spins, alphas)
    dt_psi_s = calc_dt_psi(psi_s, alphas)
    return (dt_psi_s + 1j * h_loc) / psi_s

def schrodinger_res_per_config(model, alpha, sampler, h_map, h_mat_list):
    spins = sampler(model, alpha)
    h_mult = calc_h_mult(model, alpha, h_mat_list)
    #broadcast alpha to spin shape. cannot work on view as with spins since it requires grad
    if (alpha.shape[1] == 1):
        alpha = alpha.repeat(1, spins.shape[1], 1)
    #gradient needed for dt_psi
    alpha.requires_grad = True
    
    psi_s = model.call_forward(spins, alpha)
    sp_h = get_sp(spins, h_map)
    psi_sp_h = model.call_forward_sp(sp_h, alpha)
    h_mat = torch.cat(h_mat_list, dim=3)
    h_loc = calc_Oloc(psi_sp_h, h_mat, spins, h_mult)
    dt_psi_s = calc_dt_psi(psi_s, alpha)
    return dt_psi_s + 1j * h_loc

def schrodinger_res_per_config_time_dep(model, t, h_func, sampler, h_map, h_mat_list):
    t.requires_grad = True
    h = h_func(t)
    alpha = torch.stack((t,h), dim=2)
    spins = sampler(model, alpha)
    h_mult = calc_h_mult(model, alpha, h_mat_list)
    #broadcast alpha to spin shape. cannot work on view as with spins since it requires grad
    if (alpha.shape[1] == 1):
        alpha = alpha.repeat(1, spins.shape[1], 1)
    
    psi_s = model.call_forward(spins, alpha)
    sp_h = get_sp(spins, h_map)
    psi_sp_h = model.call_forward_sp(sp_h, alpha)
    h_mat = torch.cat(h_mat_list, dim=3)
    h_loc = calc_Oloc(psi_sp_h, h_mat, spins, h_mult)
    dt_psi_s = calc_dt_psi_time_dep(psi_s, t)
    return dt_psi_s + 1j * h_loc

def abs_sq(x):
  return x.real**2 + x.imag**2

