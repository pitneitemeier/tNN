import numpy as np
from torch._C import dtype
from torch.utils.data import Dataset, DataLoader
import torch
import utils
import Operator as op

class Train_Data(Dataset):
    def __init__(self, h_list, h_ranges_list, t_min=0, t_max=1):
        assert(len(h_list) -1 == len(h_ranges_list))

        #saving ranges to generate alpha values each batch
        self.t_min = t_min
        self.t_max = t_max
        self.h_ranges_list = h_ranges_list

        #saving mat elements to pass to training loop with respective multipliers each loop
        self.num_op_h_list = [len(x) for x in h_list]
        self.num_summands_h = int(torch.tensor(self.num_op_h_list).sum())
        
    def __len__(self):
        #just setting 100000 as dataset size to get 100000 alphas for one epoch
        return 100000

    def __getitem__(self, index):
        '''
        Parameters
        ----------
        index : int
            dummy batch index
        Returns
        -------
        spins : tensor
            shape = (num_spin_configs, lattice_sites)
        alpha_arr : tensor
            shape = (num_spin_configs, num_ext_params + time)
        alpha_0 : tensor
            same as alpha_arr but with t=0 to satisfy init cond
        ext_param_scale : tensor
            tensor to scale matrix elements of hamiltonian according to external parameters
            shape = (num_spin_configs, num_summands_h, 1)
        '''
        #creating the random alpha array of numspins with one value of (t, h_ext1, ..)
        alpha_arr = torch.zeros((1, (len(self.num_op_h_list)))) 
        ext_param_scale = torch.ones((1, self.num_summands_h))
        if self.h_ranges_list is not None:
            current_ind = self.num_op_h_list[0]
            for i in range( len(self.num_op_h_list) - 1 ):
                max = self.h_ranges_list[i][1]
                min = self.h_ranges_list[i][0] 
                randval = ( (max - min) * torch.rand((1,1)) + min )
                ext_param_scale[:, current_ind : current_ind + self.num_op_h_list[i+1]] = randval
                alpha_arr[:, i+1] = randval
                current_ind += self.num_op_h_list[i+1]
        alpha_0 = alpha_arr.clone()
        alpha_arr[:, 0] = ( ( self.t_max - self.t_min ) * torch.rand((1,1)) + self.t_min )

        return alpha_arr, alpha_0, ext_param_scale


class Val_Data(Dataset):
    def __init__(self, ED_data, ext_params):
        #target Magnetizations and corresponding times from ED Code
        self.t_arr = torch.from_numpy(ED_data[0, :, 0]).reshape(-1,1,1)
        #print('t_arr shape', self.t_arr.shape)
        self.O_target = torch.from_numpy(ED_data[:, :, 1])

        self.ext_params = torch.from_numpy(ext_params)
        
    def __len__(self):
        #just full batch training here with all t
        return self.ext_params.shape[0]

    def __getitem__(self, index):
        t_arr = self.t_arr
        ext_params = self.ext_params[index,:].reshape(1, 1, -1).expand(t_arr.shape[0], 1, -1)
        #print('repeated t_arr shape ',t_arr.shape)
        #print('repeated_ext shape ', ext_params.shape)
        alpha = torch.cat((t_arr, ext_params), dim=2)
        #print(alpha[:10, :, :])
        return alpha, self.O_target[index]



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