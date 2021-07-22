import torch
print('testing wether cuda is availible')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

psi_hidden = 128
psi_type = torch.complex128
self.psi = nn.Sequential(
    utils.to_complex(),
    nn.Linear( mult_size, psi_hidden, dtype=psi_type),
    utils.odd_act(),
    nn.Linear( psi_hidden, psi_hidden, dtype=psi_type),
    utils.odd_act(),
    nn.Linear( psi_hidden, psi_hidden, dtype=psi_type),
    utils.odd_act(),
    nn.Linear( psi_hidden, 1, dtype=psi_type),
    utils.odd_act(),

)



hidden_size = 64
self.simple_model = nn.Sequential(
    nn.Linear(  lattice_sites + 2 , hidden_size, dtype=g_dtype),
    nn.ReLU(),
    utils.to_complex(),
    nn.Linear( hidden_size, hidden_size, dtype=torch.complex64),
    utils.complex_tanh(),
    nn.Linear( hidden_size, hidden_size, dtype=torch.complex64),
    utils.complex_tanh(),
    nn.Linear( hidden_size, 1, dtype=torch.complex64),
)


psi_hidden = 64
self.psi = nn.Sequential(
    nn.Linear( mult_size, psi_hidden ),
    nn.CELU(),
    nn.Linear(psi_hidden, psi_hidden),
    nn.CELU(),
    nn.Linear(psi_hidden, psi_hidden),
    nn.CELU(),
    nn.Linear(psi_hidden, 2)
)


self.tNN = nn.Sequential(
    utils.to_complex(),
    nn.Linear(2, 16, dtype=torch.complex64),
    utils.complex_CELU(),
    nn.Linear(16, mult_size, dtype=torch.complex64),
    utils.complex_CELU(),
) 


mult_size = 16 * (lattice_sites + 3)
self.lattice_net = nn.Sequential(
    nn.Conv1d(1, 8, kernel_size=2, padding=1, padding_mode='circular'),
    utils.even_act(),
    nn.Conv1d(8, 8, kernel_size=2, padding=1, padding_mode='zeros'),
    nn.CELU(),
    nn.Conv1d(8, 16, kernel_size=2, padding=1, padding_mode='zeros'),
    nn.CELU(),
    nn.Flatten(start_dim=1, end_dim=-1),
)




    #one_hot_spins = utils.flat_one_hot(spins)
    #one_hot_spins = self.flatten_spins(one_hot_spins)
    #lat_out = self.lattice_net(one_hot_spins)
    #lat_out = self.lattice_net(spins)
        #rad_and_phase = self.psi( torch.cat((t_out, lat_out), dim=1))
    #rad_and_phase = self.psi( t_out * lat_out )
    #psi = rad_and_phase[:, 0] * torch.exp( np.pi * 2j * self.sigmoid(rad_and_phase[:, 1]))
    #psi = rad_and_phase[:, 0] * torch.exp( 1.j * rad_and_phase[:, 1] ) +  rad_and_phase[:, 2] * torch.exp( -1.j * rad_and_phase[:, 3] )
    #psi = self.simple_model( torch.cat((spins, alpha), dim=1))




    lattice_hidden = 64
    mult_size = 64
    self.lattice_net = nn.Sequential(
      nn.Linear( lattice_sites, lattice_hidden, dtype=g_dtype),
      nn.CELU(),
      nn.Linear( lattice_hidden, mult_size, dtype=g_dtype),
      nn.CELU(),
      act.to_complex()
    )

    mult_size = 16 * (lattice_sites + 2)
    self.lattice_net = nn.Sequential(
      nn.Conv1d(1, 8, kernel_size=2, padding=1, padding_mode='circular', dtype=g_dtype),
      utils.even_act(),
      nn.Conv1d(8, 16, kernel_size=2, padding=1, padding_mode='zeros', dtype=g_dtype),
      nn.CELU(),
      nn.Flatten(start_dim=1, end_dim=-1),
    )