import numpy as np
from torch._C import dtype
from torch.utils.data import Dataset, DataLoader
import torch
import utils
import Operator as op

class Train_Data(Dataset):
    def __init__(self, h_param_range=None, epoch_len=100000):
        self.h_param_range = h_param_range
        #just setting fixed epoch len to define the interval for validation and lr scheduling
        self.epoch_len = int(epoch_len)
    def __len__(self):
        return self.epoch_len

    def __getitem__(self, index):
        '''
        Parameters
        ----------
        index : int
            dummy batch index
        Returns
        -------
        alpha_arr : tensor
            shape = (num_spin_configs, num_ext_params + time)
        alpha_0 : tensor
            same as alpha_arr but with t=0 to satisfy init cond
        ext_param_scale : tensor
            tensor to scale matrix elements of hamiltonian according to external parameters
            shape = (num_spin_configs, num_summands_h, 1)
        '''
        #creating the random alpha array of numspins with one value of (t, h_ext1, ..)‚
        if self.h_param_range is not None:
            alpha_arr = torch.zeros((1, (len(self.h_param_range) + 1))) 
            for i in range( len(self.h_param_range) ):
                max = self.h_param_range[i][1]
                min = self.h_param_range[i][0] 
                alpha_arr[:, i+1] = ( (max - min) * torch.rand((1,1)) + min )
        else:
            alpha_arr = torch.zeros((1,1)) 
        alpha_arr[:, 0] = torch.rand(1,1)
        return alpha_arr


class Val_Data(Dataset):
    def __init__(self, val_h_params):
        self.num_val_h_params = val_h_params.shape[0]
        
    def __len__(self):
        return self.num_val_h_params

    def __getitem__(self, index):
        return index



if (__name__ == '__main__'):
    
    ###setting up hamiltonian ###
    lattice_sites = 4
    g_dtype = torch.float64

    h2_range = [(0.15, 1.35)]

    h1 = []
    for l in range(lattice_sites):
        h1 = op.Sz(l) * (op.Sz((l+1) % lattice_sites)) + h1

    h2 = []
    for l in range(lattice_sites):
        h2 = op.Sx(l) + h2

    o = []
    for l in range(lattice_sites):
        o = op.Sx(l) * (1 / lattice_sites) + o

    h1_mat = utils.get_total_mat_els(h1, lattice_sites)
    h2_mat = utils.get_total_mat_els(h2, lattice_sites)
    o_mat = utils.get_total_mat_els(o, lattice_sites)

    h_map = utils.get_map(h1 + h2, lattice_sites)
    o_map = utils.get_map(o, lattice_sites)

    '''
    ED_data_02 = np.loadtxt('ED_data/ED_data4_02.csv', delimiter=',')
    ED_data_05 = np.loadtxt('ED_data/ED_data4_05.csv', delimiter=',')
    ED_data_07 = np.loadtxt('ED_data/ED_data4_07.csv', delimiter=',')
    ED_data_10 = np.loadtxt('ED_data/ED_data4_1.csv', delimiter=',')
    ED_data_13 = np.loadtxt('ED_data/ED_data4_13.csv', delimiter=',')

    ED_data = np.stack((ED_data_02, ED_data_05, ED_data_07, ED_data_10, ED_data_13))
    ext_params = np.array([0.2, 0.5, 0.7, 1, 1.3]).reshape(5,1)
    val_data = Val_Data(lattice_sites, ED_data, o_mat, g_dtype, ext_params)
    val_dataloader = DataLoader(val_data, batch_size=1, num_workers=24)
    val_iter = iter(val_dataloader) 
    spins, alpha, o_mat, o_target = next(val_iter)
    print('spin shape', spins.shape)
    #print(spins[0, :, :])
    print('o_mat_shape', o_mat.shape)
    print('o_target shape', o_target.shape)
    print('alpha shape', alpha.shape)
    '''
    #print(len(h1), len(h2))
    train_data = Train_Data(4, [len(h1), len(h2)], [(0.9, 1.1)], torch.float64)
    train_dataloader = DataLoader(train_data, 2)
    train_iter = iter(train_dataloader)
    alpha_arr, alpha_0, ext_param_scale = next(train_iter)
    print('ext_param_scale', ext_param_scale.shape)
    print('alpha: ', alpha_arr.shape)