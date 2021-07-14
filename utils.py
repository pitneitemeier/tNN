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
  mat_els = torch.full((1, 2, len(operator), lattice_sites), 1, dtype=torch.complex64)
  for i, op_tuple in enumerate(operator):
    for op in op_tuple:
      mat_els[:, :, i, op.lat_site] = op.mat_els
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
  spin_one_hot = torch.stack((0.5*(spin_config+1), 0.5*(1-spin_config)))
  return spin_one_hot.reshape(spin_config.shape[0], spin_config.shape[1], 2, 1, spin_config.shape[2])


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


def calc_Oloc(psi_sp, mat_els, spin_config):
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
  #multiplying with corresponding weights and summing over s' for each input configuration
  O_loc = (torch.conj(res) * psi_sp.reshape(*res.shape)).sum(2)
  return O_loc.unsqueeze(2)


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
  row = 0
  dt_psi_s = 0
  for i in range(psi_s.shape[0]):
    for j in range(psi_s.shape[1]):
      if (j==0):
        row = torch.autograd.grad(psi_s[i,j,:], alpha, create_graph=True)[0][i, j, 0].reshape(1,)
      else:
        row = torch.cat( (row, torch.autograd.grad(psi_s[i,j,:], alpha, create_graph=True)[0][i, j, 0].reshape(1,)) )
    if (i==0):
      dt_psi_s = row.reshape(1, row.shape[0])
    else:
      dt_psi_s = torch.cat((dt_psi_s, row.reshape(1, row.shape[0])), dim=0)
  return dt_psi_s.unsqueeze(2)


def loss(psi_s, psi_sp_h, psi_sp_o, dt_psi_s, o_loc, h_loc):
  #print(dt_psi_s.shape, h_loc.shape)
  h_loc_sq_sum = (torch.abs(h_loc)**2).sum(1)
  dt_psi_sq_sum = (torch.abs(dt_psi_s)**2).sum(1)
  dt_psi_h_loc_sum = (torch.abs(dt_psi_s * h_loc)**2).sum(1) 

  abs_val = (( dt_psi_sq_sum - h_loc_sq_sum )**2).sum(0)
  angle = ( dt_psi_h_loc_sum / (dt_psi_sq_sum * h_loc_sq_sum) ).sum(0)
  return (angle + abs_val).squeeze()
  