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
    res = res * ext_param_scale

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


def psi_norm_sq(psi_s):
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
  return (abs_sq(psi_s)).sum(1)

def val_loss(psi_s, o_loc, o_target):
  psi_sq_sum = (abs_sq(psi_s)).sum(1)
  psi_s_o_loc_sum = (torch.conj(psi_s) * o_loc).sum(1)
  observable = ( psi_s_o_loc_sum * (1 / psi_sq_sum) ).squeeze(1)
  loss = (abs_sq((observable - o_target))).sum(0)
  return loss, torch.real(observable)


def get_t_end(current_epoch, num_epochs, t_range, step_after = 1):
  #calculate dynamic end time and decay rate for loss
  n = int(current_epoch / step_after) + 1
  N = int (num_epochs / step_after)
  #t_end = t_range[0] + (t_range[1] - t_range[0]) * np.log(10 * n / N + 1) / np.log( 11 )
  t_end = t_range[0] + (t_range[1] - t_range[0]) * n / N 
  #t_max = self.t_max
  loss_weight = 1e-2 / (t_end/t_range[1] + 1e-2)
  return t_end, loss_weight

def tensor_to_string(alist):
    format_list = ['{:.1f}' for item in alist] 
    s = ', '.join(format_list)
    return s.format(*alist)

def get_susceptibility(magnetization, alpha):
  return torch.autograd.grad(magnetization.sum(), alpha, retain_graph=True)[0][:, :, 1].sum(1)

import matplotlib.pyplot as plt
def plot_results(name, model, magn_op, corr_op_list, t_arr, h_param_arr, ED_magn_arr, ED_susc_arr, ED_corr_arr, plot_folder):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = model.to(device)
  assert(h_param_arr.shape[0] == ED_magn_arr.shape[0] == ED_susc_arr.shape[0] == ED_corr_arr.shape[0])
  t_arr = torch.from_numpy(t_arr).type(torch.get_default_dtype())
  tot_fig, tot_ax = plt.subplots(figsize=(10,6))
  tot_ax.set_xlabel('ht')
  magn_name = magn_op[0][0].name
  corr_name = corr_op_list[0][0][0].name
  tot_ax.set_ylabel(r'$ \langle '+ magn_name +r' \rangle$', fontsize=13)
  tot_title = f'{name}, {model.lattice_sites} Spins'
  tot_ax.set_title(tot_title)
  dummy_lines = [tot_ax.plot([], [], c='black', ls='--', label='ED'), tot_ax.plot([], [], c='black', label='tNN')]
  i=0
  for h_param, ED_magn, ED_susc, ED_corr in zip(h_param_arr, ED_magn_arr, ED_susc_arr, ED_corr_arr):
      from collections import Sequence
      if isinstance(h_param, Sequence):
          h_param_tens = torch.tensor(h_param).reshape(1,1,-1).repeat(t_arr.shape[0])
      else:
          h_param_tens = torch.full((t_arr.shape[0], 1, 1), h_param)
      h_param_tens = h_param_tens.type(torch.get_default_dtype())
      alpha = torch.cat((t_arr.reshape(-1,1,1), h_param_tens), dim=2)
      alpha = alpha.repeat(1, model.spins.shape[1], 1)
      alpha.requires_grad = True
      
      magn = model.measure_observable(alpha, magn_op, model.lattice_sites)
      susc = get_susceptibility(magn, alpha)
      corr = [model.measure_observable(alpha, corr, model.lattice_sites).detach() for corr in corr_op_list]
      corr = torch.stack(corr)
      fig, ax = plt.subplots(4, 2, figsize=(10,6), sharex='col', gridspec_kw={'width_ratios': [20, 1], 'height_ratios': [2,2,1,1]})
      ax[0,1].axis('off')
      ax[1,1].axis('off')
      if isinstance(h_param, Sequence):
          label = f'$h={tensor_to_string(h_param)} h_c$'
      else:
          label = f'$h={h_param} h_c$'
      title = tot_title + ', ' + label
      ax[0,0].set_title(title)
      tot_ax.plot(t_arr, magn.detach().cpu(), label=label, c=f'C{i}')
      tot_ax.plot(t_arr, ED_magn, ls='--', c=f'C{i}')

      ax[0, 0].plot(t_arr, magn.detach().cpu(), label='tNN', c='C0')
      ax[0, 0].plot(t_arr, ED_magn, label = 'ED', ls='--', c='C1')
      ax2_1 = ax[0,0].twinx()
      ax[0, 0].set_zorder(ax2_1.get_zorder() + 1)
      ax[0, 0].patch.set_visible(False)
      ax2_1.tick_params(axis='y', labelcolor='grey')
      ax2_1.set_ylabel('Residuals', c='grey')
      ax2_1.plot(t_arr, ED_magn - magn.detach().cpu().numpy(), label='Residuals', c='grey', ls='dotted')
      ax[0,0].legend()
      ax[1, 0].plot(t_arr, susc.detach().cpu(), label='tNN', c='C0')
      ax[1, 0].plot(t_arr, ED_susc, label = 'ED', ls='--', c='C1')
      ax2_1 = ax[1,0].twinx()
      ax[1, 0].set_zorder(ax2_1.get_zorder() + 1)
      ax[1, 0].patch.set_visible(False)
      ax2_1.tick_params(axis='y', labelcolor='grey')
      ax2_1.set_ylabel('Residuals', c='grey')
      ax2_1.plot(t_arr, ED_susc - susc.detach().cpu().numpy(), label='Residuals', c='grey', ls='dotted')
      ax[1,0].legend()
      
      y = np.arange(0,model.lattice_sites/2 + 1) + 0.5
      X, Y = np.meshgrid(t_arr, y)
      c1 = ax[2,0].pcolor(X, Y, corr[:, :-1].cpu(), label='tNN')
      plt.colorbar(c1, cax=ax[2,1])
      X, Y = np.meshgrid(t_arr, y)
      c2 = ax[3,0].pcolor(X, Y, ED_corr[:, :-1], label='ED')
      plt.colorbar(c2, cax=ax[3,1])
      
      ax[2,0].set_yticks(y[:-1] + 0.5)
      ax[2,0].legend()
      ax[3,0].set_yticks(y[:-1] + 0.5)
      ax[3,0].legend()
      ax[3,0].set_xlabel('ht')
      ax[0,0].set_ylabel(r'$ \langle ' + magn_name + r' \rangle$', fontsize=13)
      ax[1,0].set_ylabel(r'$ \frac{\partial \langle ' + magn_name + r' \rangle}{\partial h}$', fontsize=17)
      ax[2,0].set_ylabel(r'd')
      ax[3,0].set_ylabel(r'd')
      
      comm_ax = fig.add_subplot(313, frame_on=False)
      comm_ax.tick_params(left=False,
          bottom=False,
          labelleft=False,
          labelbottom=False)
      comm_ax.set_ylabel(r'$\langle ' + corr_name + '_i \cdot ' + corr_name + r'_{i+d} \rangle $'+ '\n\n', fontsize=13)
      
      #fig.tight_layout()
      fig.savefig(plot_folder + title + '.png')
      plt.close(fig)
      i+=1
  tot_ax.legend()
  tot_fig.savefig(plot_folder + tot_title + '.png')
  model.to('cpu')

def abs_sq(x):
  return x.real**2 + x.imag**2