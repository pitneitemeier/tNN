import numpy as np
from torch._C import dtype
from torch.utils.data import Dataset, DataLoader
import torch
import utils

class Train_Data(Dataset):
    def __init__(self, lattice_sites, h_mat_list, h_ranges_list, o_mat, g_dtype, t_min=0, t_max=1):
        
        assert(len(h_mat_list) - 1 == len(h_ranges_list))
        #exact sampling for now
        self.spins = utils.get_all_spin_configs(lattice_sites).type(g_dtype)
        self.g_dtype = g_dtype
        #saving ranges to generate alpha values each batch
        self.t_min = t_min
        self.t_max = t_max
        self.h_ranges_list = h_ranges_list

        #saving mat elements to pass to training loop with respective multipliers each loop
        self.h_mat_list = h_mat_list
        self.o_mat = o_mat
        
    def __len__(self):
        #just setting 100000 as dataset size to get 100000 alphas for one epoch
        return 100000

    def __getitem__(self, index):
        #creating the random alpha array of numspins with one value of (t, h_ext1, ..)
        alpha_arr = torch.zeros((self.spins.shape[0], (len(self.h_mat_list))), dtype=self.g_dtype)
        if self.h_ranges_list is not None:
            for i in range( len(self.h_mat_list) - 1 ):
                max = self.h_ranges_list[i][1]
                min = self.h_ranges_list[i][0] 
                alpha_arr[:, i+1] = ( (max - min) * torch.rand((1,1)) + min )
        alpha_0 = alpha_arr.clone()
        alpha_arr[:, 0] = ( ( self.t_max - self.t_min ) * torch.rand((1,1)) + self.t_min )
        h_mat = self.h_mat_list[0]
        for i in range(len(self.h_mat_list) - 1):
            h_mat = torch.cat((h_mat, alpha_arr[0, i +1] * self.h_mat_list[i + 1]), dim=2)

        return self.spins, alpha_arr, alpha_0, h_mat, self.o_mat

class Val_Data(Dataset):
    def __init__(self, lattice_sites, ED_data, ext_params : tuple, o_mat, g_dtype):
        #exact sampling for now
        self.spins = utils.get_all_spin_configs(lattice_sites).type(g_dtype)
        #target Magnetizations from ED Code that 
        self.t_arr = torch.from_numpy(ED_data[:, 0]).type(g_dtype).unsqueeze(1)
        self.O_target = torch.from_numpy(ED_data[:, 1]).type(g_dtype)

        #saving mat elements to pass to val loop
        self.o_mat = o_mat
        if ext_params is not None:
            self.ext_params = torch.zeros((1, len(ext_params)), dtype=g_dtype)
            for i in range(len(ext_params)):
                self.ext_params[:, i] = ext_params[i]
        else:
            self.ext_params = None
        
        
    def __len__(self):
        #just full batch training here with all t
        return self.t_arr.shape[0]

    def __getitem__(self, index):
        t_arr = self.t_arr[index].repeat(self.spins.shape[0], 1)
        if self.ext_params is not None:
            ext_param = self.ext_params.repeat(self.spins.shape[0], 1)
            alpha = torch.cat((t_arr, ext_param), dim=1)
        else:
            alpha = t_arr
        return self.spins, alpha, self.o_mat, self.O_target[index]