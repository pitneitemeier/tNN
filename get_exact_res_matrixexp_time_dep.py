import torch
import numpy as np
torch.set_default_dtype(torch.float32)
device = 'cuda'
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



def get_hamiltonian_TFI(h_param_arr, lattice_sites):
  n = lattice_sites

  size = np.power(2,n)
  H_list = []
  for h_param in h_param_arr:
    H = torch.zeros((size, size), dtype=torch.complex64, device=device)
    for i in range(n):
      H += (sigma_i(sig_z, i, n) @ sigma_i(sig_z, (i+1) % n, n) + h_param * sigma_i(sig_x, i, n))
    H_list.append(H)
  return H_list, h_param_arr

def get_mean_magn_op(lattice_sites, base_op):
  n = lattice_sites
  size = np.power(2,n)
  op = torch.zeros((size, size), dtype=torch.complex64, device=device)
  for i in range(n):
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

def calc_res(H_list, h_param, magn_op, corr_op_list, psi_init, lattice_sites, t_arr):
  delta_t = t_arr[1]-t_arr[0]
  psi_t = []
  psi_current = psi_init
  for i in range(len(t_arr)):
    psi_t.append(psi_current)
    psi_current = torch.matrix_exp(-1j*delta_t*H_list[i]) @ psi_current

  magn = [torch.conj(psi.unsqueeze(0)) @ magn_op @ psi.unsqueeze(1) for psi in psi_t]
  #susc = [torch.autograd.grad(magn_val, h_param, retain_graph=True)[0] for magn_val in magn]
  #susc = torch.tensor(susc)
  magn = torch.tensor(magn)
  corr = [torch.tensor([torch.conj(psi.unsqueeze(0)) @ corr_op @ psi.unsqueeze(1) for psi in psi_t]) for corr_op in corr_op_list]
  corr = torch.stack(corr, dim=1)
  return torch.real(magn).cpu().numpy(), torch.real(corr).cpu().numpy()


import matplotlib.pyplot as plt

def plot_res(t_arr, magn, h_param_arr):
  fig, ax = plt.subplots(2,1,figsize=(10,10), sharex=True)
  ax[0].plot(t_arr, magn, label = 'magnetization', c='C0')
  ax[0].legend()
  ax[1].plot(t_arr, h_param_arr)
  ax[1].set_ylabel('h')
  ax[0].set_ylabel('magn')
  ax[1].set_xlabel('t')
  #ax.plot(data[:, 0], data[:, 1], c="red", label=r"Transverse magnetization $\langle X\rangle$", ls='--')
  fig.savefig('ED_res.png')

def pulse(t, loc, scale, frec):
    return torch.exp(-.5*(t-loc)**2/(scale**2)) * torch.sin(t * frec)

def ramp(t, t_ramp_end=1.5, h_max=2):
    return h_max * torch.clamp(t/t_ramp_end, 0, 1)

device='cuda'
lattice_sites = 10
res_folder = f'ED_data/TFI{lattice_sites}ramp/'
#psi_init = ( 1 / np.sqrt(2) * ( e_i(0,2) + e_i(1,2) ))
psi_init = e_i(0,2)
psi_init = get_init_state_kron(psi_init, lattice_sites)
magn_op = get_mean_magn_op(lattice_sites, sig_x)
corr_op_list = [get_mean_corr_op(lattice_sites, sig_z, d + 1) for d in range(int(lattice_sites/2))]
t_arr = torch.linspace(0,3,1000, device=device)
#h_param_arr = pulse(t_arr, 1.5, 0.5, 20)
h_param_arr = ramp(t_arr, t_ramp_end=1.5, h_max=1.5)
H_list, h_param_arr = get_hamiltonian_TFI(h_param_arr, lattice_sites=lattice_sites)
magn, corr = calc_res(H_list, h_param_arr, magn_op, corr_op_list, psi_init, lattice_sites, t_arr=t_arr)
plot_res(t_arr.cpu(), magn, h_param_arr.cpu())
np.savetxt(res_folder+'ED_magn.csv',magn, delimiter=',')
np.savetxt(res_folder+'alpha.csv', torch.stack((t_arr, h_param_arr), dim=1).cpu(), delimiter=',')
np.savetxt(res_folder + 'ED_corr.csv', corr, delimiter=',')
'''
h_param_list = [0.2, 0.4, 0.6, 0.8, 1., 1.3]
#h_param_list = [0.9, 1., 1.1]
#h_param_list = [1.]
magn_list = []
susc_list = []
corr_list = []
for h_param in h_param_list:
  H, h_param = get_hamiltonian_TFI(h_param, lattice_sites)
  t_arr, magn, susc, corr = calc_res(H, h_param, magn_op, corr_op_list, psi_init, lattice_sites, t_max=3)
  magn_list.append(magn)
  susc_list.append(susc)
  corr_list.append(corr)
#plot_res(t_arr, magn_list[1], susc_list[1])
#print(susc_list)


np.savetxt(res_folder + 'ED_magn_' + f'{np.min(h_param_list)}_{np.max(h_param_list)}' + '.csv', np.stack(magn_list), delimiter=',')
np.savetxt(res_folder + 'ED_susc_' + f'{np.min(h_param_list)}_{np.max(h_param_list)}' + '.csv', np.stack(susc_list), delimiter=',')
corr_stack = np.stack(corr_list)
corr_stack = corr_stack.reshape(corr_stack.shape[0], -1)

np.savetxt(res_folder + 'ED_corr_' + f'{np.min(h_param_list)}_{np.max(h_param_list)}' + '.csv', corr_stack, delimiter=',')
np.savetxt(res_folder + 't_arr_' + f'{np.min(h_param_list)}_{np.max(h_param_list)}' + '.csv', t_arr, delimiter=',')
np.savetxt(res_folder + 'h_params_' + f'{np.min(h_param_list)}_{np.max(h_param_list)}' + '.csv', h_param_list)
'''