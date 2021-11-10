import numpy as np
from torch._C import dtype
from torch.utils.data import Dataset, DataLoader
import torch
import utils
import Operator as op

class Train_Data(Dataset):
    def __init__(self, t_range, epoch_len, h_param_range=None):
        self.h_param_range = h_param_range
        self.t_range = t_range
        #just setting fixed epoch len to define the interval for validation and lr scheduling
        self.epoch_len = epoch_len

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
        alpha : tensor
            shape = (num_spin_configs, num_ext_params + time)
        '''
        #creating the random alpha array of numspins with one value of (t, h_ext1, ..)‚
        if self.h_param_range is not None:
            alpha = torch.zeros((1, (len(self.h_param_range) + 1))) 
            for i in range( len(self.h_param_range) ):
                max = self.h_param_range[i][1]
                min = self.h_param_range[i][0] 
                alpha[:, i+1] = ( (max - min) * torch.rand((1,1)) + min )
        else:
            alpha = torch.zeros((1,1)) 
        alpha[:, 0] = (self.t_range[1] - self.t_range[0]) * torch.rand(1,1) + self.t_range[0]
        return {'alpha': alpha}


class Val_Data(Dataset):
    def __init__(self, alpha, ED_data):
        self.len = alpha.shape[0]*alpha.shape[1] # #val sets * #t_val per set
        self.num_t_val = alpha.shape[1]
        self.alpha = alpha.flatten(0,1)
        self.ED_data = ED_data.flatten(0,1)
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return {'val_set_idx': int(index/self.num_t_val), 'alpha': self.alpha[index].unsqueeze(0), 'ED_data': self.ED_data[index]}

class Test_Data(Dataset):
    def __init__(self, alpha, ED_magn, ED_susc, ED_corr):
        self.len = alpha.shape[0]*alpha.shape[1] # #val sets * #t_val per set
        self.num_t_val = alpha.shape[1]
        self.alpha = alpha.flatten(0,1)
        self.ED_magn = ED_magn.flatten(0,1)
        self.ED_susc = ED_susc.flatten(0,1)
        self.ED_corr = ED_corr.flatten(0,1)

        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return {'val_set_idx': int(index/self.num_t_val), 'alpha': self.alpha[index].unsqueeze(0), 
            'ED_magn': self.ED_magn[index], 'ED_susc': self.ED_susc[index], 'ED_corr': self.ED_corr[index]}

class Test_Data_simple(Dataset):
    def __init__(self, alpha, ED_magn, ED_corr):
        self.len = alpha.shape[0]*alpha.shape[1] # #val sets * #t_val per set
        self.num_t_val = alpha.shape[1]
        self.alpha = alpha.flatten(0,1)
        self.ED_magn = ED_magn.flatten(0,1)
        self.ED_corr = ED_corr.flatten(0,1)

        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return {'val_set_idx': int(index/self.num_t_val), 'alpha': self.alpha[index].unsqueeze(0), 
            'ED_magn': self.ED_magn[index], 'ED_corr': self.ED_corr[index]}


class Simple_Data(Dataset):
    def __init__(self, alpha):
        self.len = alpha.shape[0]*alpha.shape[1] # #val sets * #t_val per set
        self.num_t_val = alpha.shape[1]
        self.alpha = alpha.flatten(0,1)
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return {'val_set_idx': int(index/self.num_t_val), 'alpha': self.alpha[index]}

class Train_Data_time_dep(Dataset):
    def __init__(self, t_range, epoch_len):
        self.t_range = t_range
        #just setting fixed epoch len to define the interval for validation and lr scheduling
        self.epoch_len = epoch_len

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, index):
        t = (self.t_range[1] - self.t_range[0]) * torch.rand(1) + self.t_range[0]
        return {'t': t}

if (__name__ == '__main__'):
   alpha = torch.cat((torch.arange(0, 10).reshape(1,-1,1).repeat(4, 1, 1), torch.arange(0,4).reshape(-1,1,1).repeat(1, 10, 1)), 2)
   dataset = Val_Data(alpha)
   dataloader = DataLoader(dataset, 13)
   dataiter = iter(dataloader)
   print(next(dataiter))