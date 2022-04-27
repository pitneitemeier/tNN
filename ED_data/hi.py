import torch
import numpy as np
device = 'cpu'
lattice_sites = 15
print(f'starting for {lattice_sites} lattice sites', flush=True)

def e_i(index, size):
  arr = torch.zeros(size, dtype=torch.complex64, device=device)
  arr[index] = 1.0
  return arr

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
print('initialized matrices', flush=True)

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
    print(f'summand of H #{i}', flush=True)
    H += (sigma_i(sig_z, i, n) @ sigma_i(sig_z, (i+1) % n, n) + h_param * sigma_i(sig_x, i, n))
  w, v = torch.linalg.eigh(H)
  return w, v, h_param

def get_mean_magn_op(lattice_sites, base_op):
  n = lattice_sites
  size = np.power(2,n)
  op = torch.zeros((size, size), dtype=torch.complex64, device=device)
  for i in range(n):
    print(f'summand of magn #{i}', flush=True)
    op += sigma_i(base_op, i, n) / n
  return op

def calc_magn(w, v, h_param, magn_op, psi_init, lattice_sites, t_min=0, t_max=1):
  t_arr = torch.linspace(t_min,t_max,50, device=device)
  magn_list = []
  for t in t_arr: 
    print(f"calculating for t={t}", flush=True)
    psi_t = v @ torch.diag(torch.exp(-1j*w*t)) @ torch.conj(torch.transpose(v, 0, 1)) @ psi_init
    magn = torch.conj(psi_t.unsqueeze(0)) @ magn_op @ psi_t.unsqueeze(1)
    magn_list.append(torch.real(magn).item())
  return t_arr.cpu().numpy(), np.array(magn_list)

def get_init_state_kron(psi_init, lattice_sites):
  temp = psi_init.clone()
  for i in range(lattice_sites-1):
    psi_init = torch.kron( temp, psi_init)
  return psi_init

import matplotlib.pyplot as plt

def plot_res(t_arr, h_arr, magn_list):
  fig, ax = plt.subplots(figsize=(10,10))
  i=0
  for h, magn in zip(h_arr, magn_list):
    ax.plot(t_arr, magn, label = f'{h:.1f}', c=f'C{i}')
    i+=1
  ax.legend()
  #ax.plot(data[:, 0], data[:, 1], c="red", label=r"Transverse magnetization $\langle X\rangle$", ls='--')
  fig.savefig('ED_res.png')


res_folder = f'TFI{lattice_sites}x/'
psi_init = ( 1 / np.sqrt(2) * ( e_i(0,2) - e_i(1,2) ))
#psi_init = e_i(0,2)
print("building initial state", flush=True)
psi_init = get_init_state_kron(psi_init, lattice_sites)
print('building magn op', flush=True)
magn_op = get_mean_magn_op(lattice_sites, sig_x)

h_param_list = [0.2, 0.4, 0.7, 1., 1.3]
magn_list = []
for h_param in h_param_list:
  print(f"starting h = {h_param:.1f}", flush=True)
  w,v, h_param = get_hamiltonian_TFI(h_param, lattice_sites)
  t_arr, magn = calc_magn(w,v, h_param, magn_op, psi_init, lattice_sites, t_max=3)
  magn_list.append(magn)

plot_res(t_arr, h_param_list, magn_list)

#saving t_arr, the parameters of the hamiltonian and the magnetization
np.savetxt(res_folder + 'ED_magn_' + f'{np.min(h_param_list)}_{np.max(h_param_list)}' + '.csv', np.stack(magn_list), delimiter=',')
np.savetxt(res_folder + 't_arr_' + f'{np.min(h_param_list)}_{np.max(h_param_list)}' + '.csv', t_arr, delimiter=',')
np.savetxt(res_folder + 'h_params_' + f'{np.min(h_param_list)}_{np.max(h_param_list)}' + '.csv', h_param_list)