#################################################################################################
To compile all the files run: make                                                      	                                                                                           	
To delete .o files run: make clean                                                      	    
                                                                                        	                                                                                       
To run an executable file 10 times and get mean execution CPUtime,GPUtime and standard  	    
deviation for all array sizes between 64 and 16384 values at once follow these steps:            
1)Create a folder with name input_files in the same directory with the script                    
2)Create a file in the previus folder with name input_size, where size is the specified          
  image width for the execution.								    
  -optimized.cu input file format:								   
   1st line: filter radius									   
   2nd line: image width									   
  -tile.cu/stream.cu input file format:						           
   1st line: filter radius									   
   2nd line: total tile size (as the number of elements that contains)			   
   3rd line: image width     								           
3)Specify the executable in the following format: sh ./script.sh ./executable_name image_width  
  *In executable optimized.cu you can follow the above steps but instead of script.sh           
   put script2.sh in order to get mean execution time for kernels and memcpys.                  
  **NOTE: in file stream.cu if CPU macro is undifined script will print an error because        
          no CPU file will be created. 							   
This script will write all the measurements in the directory named as executable_nametest.      
It will also make a file with name "log_file" with the mean execution time and standard         
deviation, following by the size of the input array [64,128,...,16384] using manipulator.c                                                                    	
                                                                                        	   
If you want to run again this script with the same executable file,                     	   
please delete the previous test file                                                    	   
with the command: rm -r executable_nametest                                             	   
#################################################################################################
