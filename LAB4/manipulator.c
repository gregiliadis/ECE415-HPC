#include<stdio.h>
#include<errno.h>
#include<string.h>
#include<stdlib.h>
#include<float.h>
#include<math.h>

int main(int argc, char *argv[]) {
	FILE *file = NULL, *log_file = NULL;
	int i,errnum;
	double array[12], max = 0, min = DBL_MAX, sum = 0, std_dev = 0, avg_time = 0;
	double *min_ptr, *max_ptr;
	//int ret;

	if(argc != 2 ) {
		printf("Wrong arguments.\n");
		exit(-1);
	}
	
	file = fopen(argv[1],"r");
	if (file == NULL) {
        	errnum = errno;
        	fprintf(stderr, "Error opening file: %s\n", strerror( errnum ));
		exit(-1);
	}
	//array initialization
	for(i = 0; i < 12; i++) {
		array[i] = -1;
	}
	//read data from file
   	for(i = 0; i < 12; i++ ) {
		fscanf(file," %lf", &array[i]);
	}
	//close file
	fclose(file);
	//find min and max
	for(i = 0; i < 12; i++) {
		if(array[i] < min) {
			min = array[i];
			min_ptr = &array[i];
		}
		if(array[i] > max) {
			max = array[i];
			max_ptr = &array[i];
		}
		sum = sum + array[i];
	}	
	log_file = fopen("log_file", "ab+");
	
	fprintf(log_file,"------%s------\n", argv[1]);
	//calculate avarage time.
	avg_time = (sum-min-max)/10;
	fprintf(log_file,"AvgRT: %lf\n", avg_time);
	//calcute standard deviation
	for(i = 0 ; i < 12; i++) {
		if( (&array[i] != min_ptr) && (&array[i] != max_ptr) ) {
			std_dev += pow(array[i]-avg_time, 2);
		}
	}
	std_dev = sqrt(std_dev/10);
	fprintf(log_file,"StdDev: %lf\n", std_dev);
	fclose(log_file);
return(0);
}
