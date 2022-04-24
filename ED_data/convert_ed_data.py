import argparse
import sys
from pathlib import Path
from shutil import rmtree

import numpy as np
import zarr


if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("dest", type=Path)
    p.add_argument("sources", type=Path, nargs="+")
    p.add_argument("-p", "--params", type=str, required=True)
    p.add_argument("--overwrite", action="store_true")

    args = p.parse_args()
    
    dest = args.dest  # type: Path
    param_names = args.params.split(",")

    if dest.exists() and not args.overwrite:
        print(f"ERROR: destination '{dest}' already exists")
        sys.exit(-1)
    elif dest.exists() and args.overwrite:
        rmtree(dest)
    for src in args.sources:
        if not (src.exists() and src.is_dir()):
            print(f"ERROR: source '{src}' is not a directory")
            sys.exit(-1)

    data = [zarr.open(src) for src in args.sources]
    pars = [[ds.attrs["system"][p] for p in param_names] for ds in data]

    dest.mkdir()
    np.savetxt(dest / "h_params.csv", pars, delimiter=",")

    times = np.array([ds.times for ds in data])
    if np.sum(np.var(times, axis=0)) > 1e-16:
        print("ERROR: Datasets were not computed at equal times")
        sys.exit(-2)
    np.savetxt(dest / "t_arr.csv", times[0], delimiter=",")

    np.savetxt(dest / "ED_Sx_mid.csv", np.array([ds.sx_mid for ds in data]), delimiter=",")
    np.savetxt(dest / "ED_magn.csv", np.array([ds.sx_avg for ds in data]), delimiter=",")

    L = data[0].attrs["system"]["L"]
    corr = np.array([[ds[f"Czz_{d}"] for d in range(L // 2)] for ds in data])
    np.savetxt(dest / "ED_corr.csv", corr.reshape(len(data), -1), delimiter=",")