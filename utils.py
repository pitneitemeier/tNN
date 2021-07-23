import torch
import numpy as np

def get_map(operator, lattice_sites):
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
  map = torch.ones((1, 1, len(operator), lattice_sites), dtype=torch.int8)
  for i, op_tuple in enumerate(operator):
    for op in op_tuple:
      map[:, :, i, op.lat_site] = op.switching_map
  return map


def get_total_mat_els(operator, lattice_sites):
#1.dim: Batch
#2.dim: two possibilities for mat el depending on input first=+1, second=-1
#3.dim: Number of summands in operator
#4.dim: number of lattice sites
  mat_els = torch.ones((1, 1, 2, len(operator), lattice_sites), dtype=torch.complex64)
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
  #print('one_hot dtpye:', spin_one_hot.dtype)
  return spin_one_hot.unsqueeze(3)

def flat_one_hot(spin_config):
  '''
  Parameters
  ----------
  spin_config : tensor
    shape = (batch, lattice_sites)
  Returns
  -------
  spin_one_hot : tensor
    shape = (batch, 2 * lattice_sites)
  '''
  spin_one_hot = torch.stack((0.5*(spin_config+1), 0.5*(1-spin_config)), dim=1)
  return spin_one_hot


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
  #print('psi_sp, mat_els, spin_config shape: ', psi_sp.shape, mat_els.shape, spin_config.shape)
  #using onehot encoding of spin config to select the correct mat_el from all mat_els table by multiplication
  s_onehot = get_one_hot(spin_config)
  #summing out zeroed values in dim 2
  #print('mat_els shape ', mat_els.shape)
  #print('s_onehot shape ', s_onehot.shape)
  res = (mat_els * s_onehot).sum(2)
  #print('res_shape', res.shape)
  #product of all matrix elements for one summand of the hamiltonian.
  res = res.prod(3)
  #print('res_shape', res.shape)

  #scaling all summands of the hamiltonian with external params
  if ext_param_scale is not None:
    #print('res_shape', res.shape)
    #print('ext_params_scale shape', ext_param_scale.shape)
    res *= ext_param_scale

  #multiplying with corresponding wave function and summing over s' for each input configuration
  #print("res, psi_sp shape: ",res.shape, psi_sp.shape)
  O_loc = (torch.conj(res.unsqueeze(3)) * psi_sp).sum(2)
  return O_loc


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


def calc_dt_psi(psi_s, alpha):
  #TODO documentation
  #print('psi in derivative:', psi_s)
  #print('dtype of psi_s.sum: ', psi_s.sum().dtype)
  dt_psi_s_real = torch.autograd.grad(torch.real(psi_s.sum()), alpha, create_graph=True)[0][:,:, 0]
  dt_psi_s_imag = torch.autograd.grad(torch.imag(psi_s.sum()), alpha, create_graph=True)[0][:,:, 0]
  dt_psi_s = dt_psi_s_real + 1.j * dt_psi_s_imag
  #print('dtype of dt_psi: ', dt_psi_s.dtype)
  return dt_psi_s.unsqueeze(2)


def psi_norm(psi_s):
  '''
  Returns the Norm of a Wave function batch wise
  Parameters
  ----------
  psi_s : tensor
    The wave function of input spins
    shape = (num_alpha, num_spins, 1)
  Returns
  -------
  norm : tensor
    shape = (num_alpha, 1)  
  '''
  return (torch.abs(psi_s)**2).sum(1)



def train_loss2(dt_psi_s, h_loc, psi_s, psi_s_0, o_loc, alpha, loss_weight):
  #part to satisfy initial condition
  psi_s_0_sq_sum = (torch.abs(psi_s_0)**2).sum(1)
  psi_0_o_loc_sum = (torch.conj(psi_s_0) * o_loc).sum(1)
  init_cond = torch.mean( (torch.abs( (psi_0_o_loc_sum / psi_s_0_sq_sum) - 1)) ** 2 )
  #print(torch.mean(psi_0_o_loc_sum / psi_s_0_sq_sum))

  #part to satisfy schrödinger equation
  h_loc_sq_sum = (torch.abs(h_loc)**2).sum(1)
  dt_psi_sq_sum = (torch.abs(dt_psi_s)**2).sum(1)
  dt_psi_h_loc_sum = (torch.conj(dt_psi_s) * h_loc).sum(1)
  #print("abs val diff: ", torch.abs(h_loc_sq_sum - dt_psi_h_loc_sum))
  schroedinger = torch.mean( torch.exp(- loss_weight * alpha[:, 0, 0]) * torch.abs( h_loc_sq_sum + dt_psi_sq_sum - 2 * torch.imag(dt_psi_h_loc_sum) ) ** 2)
  #schroedinger = torch.mean( torch.abs( h_loc_sq_sum + dt_psi_sq_sum - 2 * torch.imag(dt_psi_h_loc_sum) ) ** 2)

  #part to encourage a normed wave fun
  psi_norm_target = 1
  batched_norm = psi_norm(psi_s)
  norm = torch.mean( (batched_norm - psi_norm_target) ** 2 )

  return schroedinger + 50 * init_cond + 2 * norm , schroedinger, 50 * init_cond, 2* norm
  #return init_cond
  
def val_loss(psi_s, o_loc, o_target):
  psi_sq_sum = (torch.abs(psi_s) ** 2).sum(1)
  psi_s_o_loc_sum = (torch.conj(psi_s) * o_loc).sum(1)
  observable = ( psi_s_o_loc_sum / psi_sq_sum ).squeeze(1)
  loss = (torch.abs((observable - o_target)) ** 2).sum(0)
  return loss, torch.real(observable)

def measure_observable(psi_s, psi_sp, o_mat, spin_config, ext_param_scale = None):
  o_loc = calc_Oloc(psi_sp, o_mat, spin_config, ext_param_scale)
  psi_s_sq_sum = (torch.abs(psi_s)**2).sum(1)
  psi_s_o_loc_sum = (torch.conj(psi_s) * o_loc).sum(1)
  return psi_s_o_loc_sum / psi_s_sq_sum
 