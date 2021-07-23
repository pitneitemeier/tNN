import torch
import pytorch_lightning as pl
import utils
import matplotlib.pyplot as plt
import numpy as np
#ugly workaround -> set default tensor type
device = 'cuda'

def wave_function(Model):
  class wave_fun(Model):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

    def call_forward(self, spins, alpha):
      '''
      makes forward callable with (num_alpha_configs, num_spin_configs)
      Parameters
      ----------
      spins: tensor, dtype=float
        tensor of input spins to wave function 
        shape = (num_spin_configs, num_alpha_configs, lattice_sites)
      alpha: tensor, dtype=float
        other inputs to hamiltonian e.g. (time, ext_param) 
        shape = (num_spin_configs, num_alpha_configs, num_inputs)
      Returns
      -------
      psi: tensor, dtype=complex
        wave function for a combination of (spins, alpha) 
        size = (num_spin_configs, num_alpha_configs, 1)
      '''
      spin_shape = spins.shape
      alpha_shape = alpha.shape
      
      spins = torch.flatten(spins, end_dim=-2)
      alpha = torch.flatten(alpha, end_dim=-2)
      
      psi = self(spins, alpha)
      return psi.reshape( spin_shape[0], spin_shape[1], 1)

    def call_forward_sp(self, sprimes, alpha):
      '''
      makes forward callable with (num_alpha_configs, num_spin_configs, num_sprimes)
      Parameters
      ----------
      spins: tensor, dtype=float
        tensor of input spins to wave function 
        shape = (num_spin_configs, num_alpha_configs, num_sprimes, lattice_sites)
      alpha: tensor, dtype=float
        other inputs to hamiltonian e.g. (time, ext_param) are broadcasted to s' shape
        shape = (num_spin_configs, num_alpha_configs, num_inputs)

      Returns
      -------
      psi: tensor, dtype=complex
        wave function for a combination of (spins, alpha) 
        size = (num_spin_configs, num_alpha_configs, num_sprimes, 1)
      '''
      sprimes_shape = sprimes.shape
      alpha_shape = alpha.shape
      alpha = alpha.unsqueeze(2)
      alpha = alpha.broadcast_to(alpha_shape[0], alpha_shape[1], sprimes_shape[2], alpha_shape[2])

      sprimes = torch.flatten(sprimes, end_dim=-2)
      alpha = torch.flatten(alpha, end_dim=-2)
      
      psi = self(sprimes, alpha)
      return psi.reshape( sprimes_shape[0], sprimes_shape[1], sprimes_shape[2], 1)
  return wave_fun


class Environment(pl.LightningModule):
  def __init__(self, model, lattice_sites, h_list, obs, t_min, t_max, num_epochs, lr):
    super().__init__()

    h_tot = sum(h_list, [])
    h_mat = utils.get_total_mat_els(h_tot, lattice_sites)
    obs_mat = utils.get_total_mat_els(obs, lattice_sites)

    h_map = utils.get_map(h_tot, lattice_sites)
    obs_map = utils.get_map(obs, lattice_sites)

    self.h_map = h_map.to(device)
    self.obs_map = obs_map.to(device)
    self.h_mat = h_mat.to(device)
    self.obs_mat = obs_mat.to(device)
    self.t_min = t_min
    self.t_max = t_max
    self.num_epochs = num_epochs
    self.model = model
    self.lr = lr


  def training_step(self, batch, batch_idx):
    spins, alpha, alpha_0, ext_param_scale = batch

    t_end, loss_weight = utils.get_t_end(self.current_epoch, self.num_epochs, self.t_min, self.t_max, step_after=2)
    
    #update uniformly sampled t_val to new t range
    alpha[:, :, 0] = (t_end - self.t_min) * alpha[:, :, 0] + self.t_min
    #gradient needed for dt_psi
    alpha.requires_grad = True

    #get psi(s, alpha)
    psi_s = self.model.call_forward(spins, alpha)
    #get s' and psi(s', alpha) for h
    sp_h = utils.get_sp(spins, self.h_map)
    psi_sp_h = self.model.call_forward_sp(sp_h, alpha)
    #calc h_loc for h
    h_loc = utils.calc_Oloc(psi_sp_h, self.h_mat, spins, ext_param_scale)

    dt_psi_s = utils.calc_dt_psi(psi_s, alpha)
    
    #get s' and psi(s', alpha) for o at t=0
    sp_o = utils.get_sp(spins, self.obs_map)
    psi_sp_o = self.model.call_forward_sp(sp_o, alpha_0)
    psi_s_0 = self.model.call_forward(spins, alpha_0)
    
    #calc o_loc for o
    o_loc = utils.calc_Oloc(psi_sp_o, self.obs_mat, spins)

    #calc loss
    loss, schrodinger, init_cond, norm = utils.train_loss2(dt_psi_s, h_loc, psi_s, psi_s_0, o_loc, alpha, loss_weight)

    self.log('schrodinger', schrodinger, prog_bar=True, logger=True)
    self.log('init_cond', init_cond, prog_bar=True, logger=True)
    self.log('norm', norm, prog_bar=True, logger=True)
    self.log('train_loss', loss, logger=True)
    self.log('end time', t_end, prog_bar=True)

    return {'loss': loss}
  
  def validation_step(self, batch, batch_idx):
    spins, alpha, o_target = batch

    t_end, _ = utils.get_t_end(self.current_epoch, self.num_epochs, self.t_min, self.t_max, step_after=2)
    max_ind = torch.argmin(torch.abs(alpha[0, :, 0, 0] - t_end))
    spins = spins[0, :max_ind, :, :]
    alpha = alpha[0, :max_ind, :, :]
    o_target = o_target[0, :max_ind]

    psi_s = self.model.call_forward(spins, alpha)
    sp_o = utils.get_sp(spins, self.obs_map)
    psi_sp_o = self.model.call_forward_sp(sp_o, alpha)
    o_loc = utils.calc_Oloc(psi_sp_o, self.obs_mat, spins)
    val_loss, observable = utils.val_loss(psi_s, o_loc, o_target)
    self.log('val_loss', val_loss, prog_bar=True, logger=True)

    return {'observable' : observable, 'time': alpha[:, 0, 0], 'target': o_target, 'ext_param': alpha[0,0,1]}

  def validation_epoch_end(self, outs):
    fig, ax = plt.subplots(figsize=(10,6))
    for i, out in enumerate(outs):
      ext_param = float(out['ext_param'].cpu())
      ax.plot(out['time'].cpu(), out['target'].cpu(), label=f'target h={ext_param:.1f}', c=f'C{i}', ls='--')
      ax.plot(out['time'].cpu(), out['observable'].cpu(), label=f'model_pred h={ext_param}', c=f'C{i}')
    ax.set_xlabel('ht')
    ax.set_ylabel('$E\,[\sigma_x]$')
    ax.legend()
    ax.set_title('TFI magnetization')
    self.logger.experiment.add_figure('magnetization', fig)
    fig.savefig('magnetization.png')
    plt.close(fig)
  
  def configure_optimizers(self):
    #optimizer = torch.optim.LBFGS(self.parameters(), lr=1e-2)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    return optimizer