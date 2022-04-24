#!/usr/bin/env fish

set L 10
set PBC 0
set Ks 0.240 0.245 0.250 0.255 0.260

for K in $Ks
    python3 tfi_xx_ed.py -length $L -g 1.00 -h 0.00 -k $K -pbc $PBC -out tfi_xx2_L{$L}_k{$K}.zarr
end

if [ $PBC -eq 1 ]
set SUFFIX _pbc
else
set SUFFIX ""
end

python3 convert_ed_data.py -p k TFI_XX2_{$L}{$SUFFIX} tfi_xx2_L{$L}_k{$Ks}.zarr --overwrite
