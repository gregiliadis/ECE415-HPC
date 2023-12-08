#!/bin/sh
#Executing app 12 times.
SIZE=64
mkdir "$1test"
for i in 1 2 3 4 5 6 7 8 9
do 
  echo $SIZE
  for j in 1 2 3 4 5 6 7 8 9 10 11 12
    do 
      $1 < "input_files/input_$SIZE"
  done > "$1test/out_$SIZE"
  SIZE=$(( SIZE*2 ))
done

SIZE=64
compare1="CPU_time"
compare2="GPU_time"
for i in 1 2 3 4 5 6 7 8 9
do
  input="$1test/out_$SIZE"
  while IFS=' ' read -r F1 F2 F3 F4
  do
    if [ "$F1" = "$compare1" ]
    then
      echo $F3 >> "$1test/CPUtime_$SIZE"
    fi
    if [ "$F1" = "$compare2" ]
    then
      echo $F3 >> "$1test/GPUtime_$SIZE"
    fi
  done < "$input"
  rm "$1test/out_$SIZE"
  SIZE=$(( SIZE*2 ))
done

SIZE=64
for i in 1 2 3 4 5 6 7 8 9
do
  ./manipulator "$1test/CPUtime_$SIZE"
  ./manipulator "$1test/GPUtime_$SIZE"
 SIZE=$((SIZE*2 ))
done
 
