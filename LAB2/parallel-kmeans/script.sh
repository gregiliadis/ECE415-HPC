#!/bin/sh
#Executing app 12 times.
ARG=" -o -b -i"
NTH=1
for i in 1 2 3 4 5 6 7
do 
  echo $NTH
  export OMP_NUM_THREADS=$NTH
  for j in 1 2 3 4 5 6 7 8 9 10 11 12
    do 
      ./seq_main $ARG Image_data/texture17695.bin "-n" $1
  done > "test/out_$1_$NTH"
  NTH=$(( NTH*2 ))
done

NTH=1
compare="Computation"
for i in 1 2 3 4 5 6 7
do
  input="test/out_$1_$NTH"
  while IFS=' ' read -r F1 F2 F3 F4 F5
  do
    if [ "$F1" = "$compare" ]
    then
      echo $F4 >> "test/time_$1_$NTH"
    fi
  done < "$input"
  rm "test/out_$1_$NTH"
  NTH=$(( NTH*2 ))
done

NTH=1
for i in 1 2 3 4 5 6 7
do
  ./manipulator "test/time_$1_$NTH"
 NTH=$(( NTH*2 ))
done
 
