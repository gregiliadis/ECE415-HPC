#!/bin/sh
#Executing app 12 times.
for i in 1 2 3 4 5 6 7 8 9 10 11 12
  do 
    ./$1
done > "test/out_$1"

flag=0
compare="Total"

input="test/out_$1"
while IFS=' ' read -r F1 F2 F3 F4 F5 
do
  if [ "$F1" = "$compare" ]
  then
     echo $F4 >> "test/time_$1"
  fi
done < "$input"
rm "test/out_$1"
