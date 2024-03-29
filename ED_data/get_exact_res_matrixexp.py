import torch
import numpy as np
device = 'cpu'
def e_i(index, size):
  arr = torch.zeros(size, dtype=torch.complex64, device=device)
  arr[index] = 1.0
  return arr
print('starting')
#Defining Pauli Matrices 

sig_z = torch.eye(2, dtype=torch.complex64, device=device)
sig_z[1,1] = -1
#print("\sigma_z = \n", sig_z)

sig_x = torch.zeros((2,2), dtype=torch.complex64, device=device)
sig_x[1,0] = 1
sig_x[0,1] = 1
#print("\sigma_x = \n", sig_x)

sig_y = torch.zeros((2,2), dtype=torch.complex64, device=device)
sig_y[1,0] = 1j
sig_y[0,1] = -1j
print('initialized matrices')
"""Defining Tensorized sigma_i"""

def sigma_i(matrix, index, system_size):
  
  if (index%system_size > 0):
    sigma_i = torch.eye(2, dtype=torch.complex64, device=device)
    for i in torch.arange(1, index%system_size):
      sigma_i = torch.kron(torch.eye(2, dtype=torch.complex64, device=device), sigma_i)
    sigma_i = torch.kron(matrix, sigma_i)

  else:
    sigma_i = matrix

  for i in torch.arange((index%system_size+1), system_size):
    sigma_i = torch.kron(torch.eye(2, dtype=torch.complex64, device=device), sigma_i)
  
  return sigma_i



def get_hamiltonian_TFI(h_param, lattice_sites):
  n = lattice_sites
  h_param = torch.full((1,), h_param, requires_grad=False, device=device)
  size = np.power(2,n)
  H = torch.zeros((size, size), dtype=torch.complex64, device=device)
  for i in range(n):
    print(f'summand of H #{i}')
    H += (sigma_i(sig_z, i, n) @ sigma_i(sig_z, (i+1) % n, n) + h_param * sigma_i(sig_x, i, n))
  return H, h_param

def get_hamiltonian_XXZ(delta, h, lattice_sites):
  #TODO
  n = lattice_sites

  h = torch.full((1,), h, requires_grad=True, device=device)
  delta = torch.full((1,), delta)
  size = np.power(2,n)
  H = torch.zeros((size, size), dtype=torch.complex64, device=device)
  for i in range(n):
    H += -(sigma_i(sig_x, i, n) @ sigma_i(sig_x, (i+1) % n, n)) \
      -(sigma_i(sig_y, i, n) @ sigma_i(sig_y, (i+1) % n, n)) \
      - delta *(sigma_i(sig_z, i, n) @ sigma_i(sig_z, (i+1) % n, n)) \
         - h * sigma_i(sig_x, i, n)
  return H, h_param

def get_mean_magn_op(lattice_sites, base_op):
  n = lattice_sites
  size = np.power(2,n)
  op = torch.zeros((size, size), dtype=torch.complex64, device=device)
  for i in range(n):
    print(f'summand of magn #{i}')
    op += sigma_i(base_op, i, n) / n
  return op

def get_mean_corr_op(lattice_sites, base_op, distance):
  n = lattice_sites
  size = np.power(2,n)
  op = torch.zeros((size, size), dtype=torch.complex64, device=device)
  for i in range(n):
    op += sigma_i(base_op, i, n) @ sigma_i(base_op, (i + distance) % n, n) / n
  return op

def get_init_state_kron(psi_init, lattice_sites):
  temp = psi_init.clone()
  for i in range(lattice_sites-1):
    psi_init = torch.kron( temp, psi_init)
  return psi_init

def get_init_state_half(lattice_sites):
  x_up = 1 / np.sqrt(2) * ( e_i(0,2) + e_i(1,2) )
  x_down = 1 / np.sqrt(2) * ( e_i(0,2) - e_i(1,2) )
  psi_init = x_down
  #for i in range(int(lattice_sites/2)-1):
  #  psi_init = torch.kron( x_up, psi_init)
  for i in range(int(lattice_sites)-1):
    psi_init = torch.kron( x_up, psi_init)
  return psi_init

def calc_res(H, h_param, magn_op, corr_op_list, psi_init, lattice_sites, t_min=0, t_max=1):
  t_arr = torch.linspace(t_min,t_max,100, device=device)
  psi_t = [torch.matrix_exp(-1j*t*H) @ psi_init for t in t_arr]

  magn = [torch.conj(psi.unsqueeze(0)) @ magn_op @ psi.unsqueeze(1) for psi in psi_t]
  susc = [torch.autograd.grad(magn_val, h_param, retain_graph=True)[0] for magn_val in magn]
  susc = torch.tensor(susc)
  magn = torch.tensor(magn)
  corr = [torch.tensor([torch.conj(psi.unsqueeze(0)) @ corr_op @ psi.unsqueeze(1) for psi in psi_t]) for corr_op in corr_op_list]
  corr = torch.stack(corr, dim=1)
  return t_arr.cpu().numpy(), torch.real(magn).cpu().numpy(), (susc).cpu().numpy(), torch.real(corr).cpu().numpy()

def calc_magn(H, h_param, magn_op, psi_init, lattice_sites, t_min=0, t_max=1):
  t_arr = torch.linspace(t_min,t_max,50, device=device)
  magn_list = []
  for t in t_arr: 
    psi_t = torch.matrix_exp(-1j*t*H) @ psi_init
    magn = torch.conj(psi_t.unsqueeze(0)) @ magn_op @ psi_t.unsqueeze(1)
    magn_list.append(torch.real(magn).item())
  return t_arr.cpu().numpy(), np.array(magn_list)

  


import matplotlib.pyplot as plt

def plot_res(t_arr, h_arr, magn_list):
  fig, ax = plt.subplots(figsize=(10,10))
  ax.set_title(f'{lattice_sites}')
  i=0
  for h, magn in zip(h_arr, magn_list):
    ax.plot(t_arr, magn, label = f'{h:.1f}', c=f'C{i}')
    i+=1
  ax.legend()
  #ax.plot(data[:, 0], data[:, 1], c="red", label=r"Transverse magnetization $\langle X\rangle$", ls='--')
  fig.savefig('ED_res.png')

lattice_sites = 12
res_folder = f'TFI{lattice_sites}x/'
psi_init = ( 1 / np.sqrt(2) * ( e_i(0,2) + e_i(1,2) ))
#psi_init = e_i(0,2)
psi_init = get_init_state_kron(psi_init, lattice_sites)
print('building magn op')
magn_op = get_mean_magn_op(lattice_sites, sig_x)
#z_magn_op = get_mean_magn_op(lattice_sites, sig_z)
#magn_op_single = sigma_i(sig_x, 0, lattice_sites)
#corr_op_list = [get_mean_corr_op(lattice_sites, sig_z, d + 1) for d in range(int(lattice_sites/2))]
h_param_list = [0.2, 0.4, 0.7, 1., 1.3]
#h_param_list = [0., 0.5, 1., 1.5, 2.]
#h_param_list = [0.9, 1., 1.1]
#h_param_list = [1.]
magn_list = []
#susc_list = []
#corr_list = []
for h_param in h_param_list:
  print(f"starting h = {h_param:.1f}")
  H, h_param = get_hamiltonian_TFI(h_param, lattice_sites)
  #t_arr, magn, susc, corr = calc_res(H, h_param, magn_op, corr_op_list, psi_init, lattice_sites, t_max=3)
  t_arr, magn = calc_magn(H, h_param, magn_op, psi_init, lattice_sites, t_max=3)
  magn_list.append(magn)
  #susc_list.append(susc)
  #corr_list.append(corr)
  #print(corr.shape)

plot_res(t_arr, h_param_list, magn_list)
#print(susc_list)


np.savetxt(res_folder + 'ED_magn_' + f'{np.min(h_param_list)}_{np.max(h_param_list)}' + '.csv', np.stack(magn_list), delimiter=',')
#np.savetxt(res_folder + 'ED_susc_' + f'{np.min(h_param_list)}_{np.max(h_param_list)}' + '.csv', np.stack(susc_list), delimiter=',')
#corr_stack = np.stack(corr_list)
#corr_stack = corr_stack.reshape(corr_stack.shape[0], -1)
#print(corr_stack.shape)

#np.savetxt(res_folder + 'ED_corr_' + f'{np.min(h_param_list)}_{np.max(h_param_list)}' + '.csv', corr_stack, delimiter=',')
np.savetxt(res_folder + 't_arr_' + f'{np.min(h_param_list)}_{np.max(h_param_list)}' + '.csv', t_arr, delimiter=',')
np.savetxt(res_folder + 'h_params_' + f'{np.min(h_param_list)}_{np.max(h_param_list)}' + '.csv', h_param_list)
