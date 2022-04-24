import argparse
from typing import Sequence, Union

import netket as nk
import numpy as np
import qutip
from qutip.ui.progressbar import EnhancedTextProgressBar
import zarr


def TFIXZHamiltonian(hi, graph, g: float, h: float, k: float):
    N = graph.n_nodes

    Z = sum(nk.operator.spin.sigmaz(hi, i) for i in range(N))
    X = sum(nk.operator.spin.sigmax(hi, i) for i in range(N))

    ZZ = sum(
        nk.operator.spin.sigmaz(hi, i) * nk.operator.spin.sigmaz(hi, j)
        for (i, j) in graph.edges()
    )
    XX = sum(
        nk.operator.spin.sigmax(hi, i) * nk.operator.spin.sigmax(hi, j)
        for (i, j) in graph.edges()
    )

    return -1.0 * (ZZ + k * XX + g * X + h * Z)


class TFIXZSystem:
    def __init__(self, n_sites: int, pbc: Union[bool, Sequence[bool]]):
        self.hi = nk.hilbert.Spin(1 / 2, N=n_sites)
        self.g = nk.graph.Chain(n_sites, pbc=pbc)

    def hamiltonian(self, g: float, h: float, k: float):
        ham = TFIXZHamiltonian(self.hi, self.g, g, h, k)
        return ham

    def sx_mid(self):
        return nk.operator.spin.sigmax(self.hi, self.L // 2 - 1)

    def sx_avg(self):
        return sum(nk.operator.spin.sigmax(self.hi, i) for i in range(self.L)) / self.L

    def avg_corr_op(self, sigma, dist: float):
        op = {"x": nk.operator.spin.sigmax, "z": nk.operator.spin.sigmaz}[sigma]
        return (
            sum(
                op(self.hi, i) + op(self.hi, (i + dist) % self.L) for i in range(self.L)
            )
            / self.L
        )

    def avg_corr_ops(self, sigma):
        return {
            f"C{sigma}{sigma}_{d}": self.avg_corr_op(sigma, d + 1)
            for d in range(syst.L // 2)
        }

    @property
    def L(self):
        return self.hi.size


def zero_or_one_to_bool(input: str):
    if input == "0":
        return False
    elif input == "1":
        return True
    else:
        raise argparse.ArgumentError("must be '0' or '1'")


def segments(t0: float, tsteps: int, dt: float, size: int = 1000, i=0):
    tcur = t0 + i * size * dt
    if tsteps <= size:
        tmax = t0 + (i * size + tsteps) * dt
        yield (i, tcur, tmax, (tcur, tmax, tsteps + 1))
    else:
        tmax = t0 + (i + 1) * size * dt
        yield (i, tcur, tmax, (tcur, tmax, size + 1))
        yield from segments(t0, tsteps - size, dt, size, i + 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-out", type=str, required=True)
    p.add_argument("-length", type=int, required=True)
    p.add_argument("-g", type=float, required=True)
    p.add_argument("-h", type=float, required=True)
    p.add_argument("-k", type=float, required=True)
    p.add_argument("-pbc", type=zero_or_one_to_bool, required=True)

    args = p.parse_args()

    syst = TFIXZSystem(args.length, args.pbc)
    ham = syst.hamiltonian(args.g, args.h, args.k).to_qobj()
    sx_mid = syst.sx_mid().to_qobj()
    sx_avg = syst.sx_avg().to_qobj()
    corr_ops = {k: v.to_qobj() for k, v in syst.avg_corr_ops(sigma="z").items()}

    # x polarized product state
    xpol = (1.0 / np.sqrt(2.0)) * (qutip.basis(2, 0) + qutip.basis(2, 1))
    psi0 = qutip.tensor(syst.L * [xpol])

    root = zarr.group(args.out)
    root.attrs["system"] = {
        "L": args.length,
        "pbc": args.pbc,
        "g": args.g,
        "h": args.h,
        "k": args.k,
    }

    trange = (0.0, 3.0, 300)
    n_times = trange[2]
    dt = trange[1] / n_times

    shape = (n_times + 1,)
    times_out = root.create("times", shape=shape, dtype=np.float64)
    sx_mid_out = root.create("sx_mid", shape=shape, dtype=np.float64)
    sx_avg_out = root.create("sx_avg", shape=shape, dtype=np.float64)
    corr_out = {
        key: root.create(key, shape=shape, dtype=np.float64) for key in corr_ops.keys()
    }

    shape = (n_times + 1, syst.hi.n_states)
    states_out = root.create("states", shape=shape, dtype=np.complex128)

    times_out[0] = trange[0]
    states_out[0, :] = np.asarray(psi0).squeeze()
    sx_mid_out[0] = qutip.expect(sx_mid, psi0)
    sx_avg_out[0] = qutip.expect(sx_avg, psi0)
    for key, op in corr_ops.items():
        corr_out[key][0] = qutip.expect(op, psi0)

    block_size = 10
    for (i, tcur, tmax, teval) in segments(trange[0], trange[2], dt, size=block_size):
        print(f"t={tcur:.2f}->{tmax:.2f} [{teval}]")
        times = np.linspace(*teval)
        print(times)
        sol = qutip.sesolve(ham, psi0, times, progress_bar=EnhancedTextProgressBar())

        cur_slice = slice(block_size * i + 1, block_size * (i + 1) + 1)
        times_out[cur_slice] = times[1:]
        states_out[cur_slice, :] = np.asarray(sol.states[1:]).squeeze()
        sx_mid_out[cur_slice] = qutip.expect(sx_mid, sol.states)[1:]
        sx_avg_out[cur_slice] = qutip.expect(sx_avg, sol.states)[1:]
        for key, op in corr_ops.items():
            corr_out[key][cur_slice] = qutip.expect(op, sol.states)[1:]

        psi0 = sol.states[-1]
        print(f"t={sol.times[-1]}")
