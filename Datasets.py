import numpy as np
from torch._C import dtype
from torch.utils.data import Dataset, DataLoader
import torch
import utils
import Operator as op

class Train_Data(Dataset):
    def __init__(self, t_range, h_param_range=None, epoch_len=100000):
        self.h_param_range = h_param_range
        self.t_range = t_range
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
        alpha_arr[:, 0] = (self.t_range[1] - self.t_range[0]) * torch.rand(1,1) + self.t_range[0]
        return alpha_arr


class Val_Data(Dataset):
    def __init__(self, val_h_params):
        self.num_val_h_params = val_h_params.shape[0]
        
    def __len__(self):
        return self.num_val_h_params

    def __getitem__(self, index):
        return index



if (__name__ == '__main__'):
   pass 