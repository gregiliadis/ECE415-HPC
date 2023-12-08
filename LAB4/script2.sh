#!/bin/sh
#Executing app 12 times.
mkdir "$1test"

for j in 1 2 3 4 5 6 7 8 9 10 11 12
    do 
      $1 < "input_files/input_$2"
done > "$1test/out_$2"
 
compare2="GPU_time"
compare3="Kernels:"
compare4="Memcpysum:"

input="$1test/out_$2"
while IFS=' ' read -r F1 F2 F3 F4
  do
    if [ "$F1" = "$compare2" ]
    then
      echo $F3 >> "$1test/GPUtime_$2"
    fi
    if [ "$F1" = "$compare3" ]
    then
      echo $F2 >> "$1test/GPUKernels_time_$2"
    fi
    if [ "$F1" = "$compare4" ]
    then
      echo $F2 >> "$1test/GPUMemcpy_time_$2"
    fi
done < "$input"
rm "$1test/out_$2"
./manipulator "$1test/GPUtime_$2"
./manipulator "$1test/GPUKernels_time_$2"
./manipulator "$1test/GPUMemcpy_time_$2"
 
