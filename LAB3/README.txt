#########################################################################################
To compile all the files run: make                                                      #
                                                                                        #
To delete .o files run: make clean                                                      #
                                                                                        #
                                                                                        #
To run an executable file 10 times and get mean execution CPUtime,GPUtime and standard  #
deviation for all array sizes between 64 and 16384 values at once follow these steps:   #
1)Specify the executable in the following format: ./script.sh ./executable_name         #
2)This script will write all the measurements in the directory named as                 #
executable_nametest. It will                                                            #
also make a file with name "log_file" with the mean execution time and standard         #
deviation, following by the size of the input array [64,128,...,16384]                  #
using manipulator.c                                                                     #
                                                                                        #
If you want to run again this script, please delete the previous test file              #
with the command: rm -r executable_nametest                                             #
#########################################################################################
