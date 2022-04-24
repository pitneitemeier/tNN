#!/usr/bin/env fish

set L 4
set PBC 0
set Ks 0.05 0.10 0.15 0.20 0.25 0.30

for K in $Ks
    python3 tfi_xx_ed.py -length $L -g 1.00 -h 0.00 -k $K -pbc $PBC -out tfi_xx_L{$L}_k{$K}.zarr
end

if [ $PBC -eq 1 ]
set SUFFIX _pbc
else
set SUFFIX ""
end

python3 convert_ed_data.py -p k TFI_XX_{$L}{$SUFFIX} tfi_xx_L{$L}_k{$Ks}.zarr --overwrite
