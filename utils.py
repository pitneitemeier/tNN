import torch
import numpy as np

def get_map(hamiltonian, lattice_sites):
#1.dim: Batch
#2.dim: Number of summands in Hamilton operator
#3.dim: number of lattice sites
  map = torch.ones((1, len(hamiltonian), lattice_sites), dtype=torch.int8)
  for i, op_tuple in enumerate(hamiltonian):
    for op in op_tuple:
      map[:, i, op.lat_site] = op.switching_map
  return map


def get_total_mat_els(hamiltonian, lattice_sites):
#1.dim: Batch
#2.dim: two possibilities for mat el depending on input first=+1, second=-1
#3.dim: Number of summands in Hamilton operator
#4.dim: number of lattice sites
  mat_els = torch.full((1, 2, len(hamiltonian), lattice_sites), 1, dtype=torch.complex64)
  for i, op_tuple in enumerate(hamiltonian):
    for op in op_tuple:
      mat_els[:, :, i, op.lat_site] = op.mat_els
  return mat_els


def get_one_hot(spin_config):
#1.dim: Batch
#2.dim: two possibilities for mat el depending on input: first=+1, second=-1
#3.dim: Number of summands in Hamilton operator
#4.dim: number of lattice sites
  return torch.stack((0.5*(spin_config+1), 0.5*(1-spin_config))).reshape(spin_config.shape[0], 2, 1, spin_config.shape[1])


#getting s_primes for O_loc via map
def get_sp(spin_config, map):
  #returns:
  #1. dim: batch size
  #2. dim: number of summands in h = number of sp configs per s_config
  #3. dim: lattice_sites
  return map*spin_config.reshape(spin_config.shape[0], 1, spin_config.shape[1])


def calc_Oloc(psi_sp, mat_els, spin_config):
  #using onehot encoding of spin config to select the correct mat_el from all mat_els table by multiplication
  s_onehot = get_one_hot(spin_config)
  #summing out zeroed values
  res = (mat_els * s_onehot).sum(1)
  #product of all matrix elements for one summand of the hamiltonian.
  res = res.prod(2)
  #multiplying with corresponding weights and summing over s' for each input configuration
  res = (torch.conj(res) * psi_sp).sum(1)
  return res


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



  