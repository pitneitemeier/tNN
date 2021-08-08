import operator as op
def avg_correlation(base_op, distance, lattice_sites):
    obs = []
    for i in range(lattice_sites):
        obs = base_op(i) * (1 / lattice_sites) * base_op((i + distance) % lattice_sites) + obs
    return obs

def avg_magnetization(base_op, lattice_sites):
    obs = []
    for l in range(lattice_sites):
        obs = base_op(l) * (1 / lattice_sites) + obs
    return obs
