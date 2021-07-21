import numpy as np
from torch._C import dtype
from torch.utils.data import Dataset, DataLoader
import torch
import utils
import Operator as op

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
        alpha_arr = torch.full((self.spins.shape[0], (len(self.h_mat_list))), 0, dtype=self.g_dtype) 
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
    def __init__(self, lattice_sites, ED_data, o_mat, g_dtype, ext_params = None):
        #exact sampling for now
        self.spins = utils.get_all_spin_configs(lattice_sites).type(g_dtype)
        #target Magnetizations from ED Code that 
        self.t_arr = torch.from_numpy(ED_data[0, :, 0]).type(g_dtype).unsqueeze(1)
        #print('t_arr shape', self.t_arr.shape)
        self.O_target = torch.from_numpy(ED_data[:, :, 1]).type(g_dtype)

        #saving mat elements to pass to val loop
        self.o_mat = o_mat
        self.ext_params = torch.from_numpy(ext_params).type(g_dtype)
        
        
    def __len__(self):
        #just full batch training here with all t
        return self.ext_params.shape[0]

    def __getitem__(self, index):
        t_arr = self.t_arr.repeat(1, self.spins.shape[0]).unsqueeze(2)
        if self.ext_params is not None:
            ext_params = self.ext_params[index,:].broadcast_to(t_arr.shape[0], self.spins.shape[0], 1)
            #print('repeated t_arr shape ',t_arr.shape)
            #print('repeated_ext shape ', ext_params.shape)
            alpha = torch.cat((t_arr, ext_params), dim=2)
        else:
            alpha = t_arr
        spins = self.spins.unsqueeze(0).repeat(self.t_arr.shape[0], 1, 1)
        return spins, alpha, self.o_mat, self.O_target[index]



if (__name__ == '__main__'):
    ###setting up hamiltonian ###
    lattice_sites = 3
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