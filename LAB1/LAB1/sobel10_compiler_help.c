// This will apply the sobel filter and return the PSNR between the golden sobel and the produced sobel
// sobelized image
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

#define SIZE	4096
#define INPUT_FILE	"input.grey"
#define OUTPUT_FILE	"output_sobel.grey"
#define GOLDEN_FILE	"golden.grey"

/* The horizontal and vertical operators to be used in the sobel filter */
char horiz_operator[3][3] = {{-1, 0, 1}, 
                             {-2, 0, 2}, 
                             {-1, 0, 1}};
char vert_operator[3][3] = {{1, 2, 1}, 
                            {0, 0, 0}, 
                            {-1, -2, -1}};

double sobel(unsigned char *input, unsigned char *output, unsigned char *golden);
int convolution2D(int posy, int posx, const unsigned char *input, char operator[][3]);

/* The arrays holding the input image, the output image and the output used *
 * as golden standard. The luminosity (intensity) of each pixel in the      *
 * grayscale image is represented by a value between 0 and 255 (an unsigned *
 * character). The arrays (and the files) contain these values in row-major *
 * order (element after element within each row and row after row. 			*/
unsigned char input[SIZE*SIZE], output[SIZE*SIZE], golden[SIZE*SIZE];


/* Implement a 2D convolution of the matrix with the operator */
/* posy and posx correspond to the vertical and horizontal disposition of the *
 * pixel we process in the original image, input is the input image and       *
 * operator the operator we apply (horizontal or vertical). The function ret. *
 * value is the convolution of the operator with the neighboring pixels of the*
 * pixel we process.														  */
int convolution2D(int posy, int posx, const unsigned char *input, char operator[][3]) {
	int i, j, res;
  
	res = 0;
	
	res += input[(posy -1)*SIZE + posx -1] * operator[0][0];
	res += input[(posy -1)*SIZE + posx ] * operator[0][1];
	res += input[(posy -1)*SIZE + posx + 1] * operator[0][2];
	res += input[(posy)*SIZE + posx -1] * operator[1][0];
	res += input[(posy)*SIZE + posx ] * operator[1][1];
	res += input[(posy)*SIZE + posx + 1] * operator[1][2];
	res += input[(posy + 1)*SIZE + posx -1] * operator[2][0];
	res += input[(posy + 1)*SIZE + posx ] * operator[2][1];
	res += input[(posy + 1)*SIZE + posx + 1] * operator[2][2];
	
	return(res);
}


/* The main computational function of the program. The input, output and *
 * golden arguments are pointers to the arrays used to store the input   *
 * image, the output produced by the algorithm and the output used as    *
 * golden standard for the comparisons.									 */
double sobel(unsigned char *restrict input, unsigned char *restrict output, unsigned char *restrict golden)
{
	register double PSNR = 0, t;
	register int i, j, k, resx, resy, t1, t2, t3, t4, t5, j1, j2, j3;
	unsigned int p;
    int sqr[65026];
	int res;
	struct timespec  tv1, tv2;
	FILE *f_in, *f_out, *f_golden;

    
    for(i = 0; i < 65026; i++){
        sqr[i] = sqrt(i);
    }
	/* The first and last row of the output array, as well as the first  *
     * and last element of each column are not going to be filled by the *
     * algorithm, therefore make sure to initialize them with 0s.		 */
	memset(output, 0, SIZE*sizeof(unsigned char));
	memset(&output[SIZE*(SIZE-1)], 0, SIZE*sizeof(unsigned char));
	for (i = 1; i < SIZE-1; i++) {
		output[i*SIZE] = 0;
		output[i*SIZE + SIZE - 1] = 0;
	}

	/* Open the input, output, golden files, read the input and golden    *
     * and store them to the corresponding arrays.						  */
	f_in = fopen(INPUT_FILE, "r");
	if (f_in == NULL) {
		printf("File " INPUT_FILE " not found\n");
		exit(1);
	}
  
	f_out = fopen(OUTPUT_FILE, "wb");
	if (f_out == NULL) {
		printf("File " OUTPUT_FILE " could not be created\n");
		fclose(f_in);
		exit(1);
	}  
  
	f_golden = fopen(GOLDEN_FILE, "r");
	if (f_golden == NULL) {
		printf("File " GOLDEN_FILE " not found\n");
		fclose(f_in);
		fclose(f_out);
		exit(1);
	}    

	fread(input, sizeof(unsigned char), SIZE*SIZE, f_in);
	fread(golden, sizeof(unsigned char), SIZE*SIZE, f_golden);
	fclose(f_in);
	fclose(f_golden);
  
	/* This is the main computation. Get the starting time. */
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
	/* For each pixel of the output image */
	//strength reduction
    t1 = 0;
    t2 = SIZE;
    t3 = SIZE << 1;
	t4 = (SIZE-2)>>1;
    t5 = (SIZE-2) % 2;
	
	for (i=1; i<SIZE-1; i+=1) {
		for (j=1 , k = 1; k <=t4 ;k++, j+=2 ) {
			j1 = t1 + j;
            j2 = t2 + j;
            j3 = t3 + j;
			//////////////////////////////////////////////////////////////////////
			resx = 0;
			resy =0;
			/* Apply the sobel filter and calculate the magnitude *
			 * of the derivative.								  */
			resx -= input[j1 -1];
			resy += input[j1 -1];
			resx += input[j1 + 1];
			resy += input[j1 + 1];
			resy += input[j1] << 1;
			resx -= input[j2 + -1] << 1;
			resx += input[j2 + 1] << 1;
			resx -= input[j3 -1];
			resy -= input[j3 -1];
			resx += input[j3 + 1];
			resy -= input[j3 + 1];
			resy -= input[j3] << 1;
			
            p = resx*resx + resy*resy;
			/* If the resulting value is greater than 255, clip it *
			 * to 255.
										   */
			if (p > 65025)
				output[j2] = 255;      
			else
				output[j2] = (unsigned char)sqr[p];
			//////////////////////////////////////////////////////////////////
			
			//////////////////////////////////////////////////////////////////////
			resx = 0;
			resy =0;
			/* Apply the sobel filter and calculate the magnitude *
			 * of the derivative.								  */
			resx -= input[j1];
			resy += input[j1];
			resy += input[j1+1] << 1;
			resx += input[j1 + 2];
			resy += input[j1+ 2];
			resx -= input[j2] << 1;
			resx += input[j2 +2] << 1;
			resx -= input[j3];
			resy -= input[j3];
			resy -= input[j3 +1] << 1;
			resx += input[j3 + 2];
        	resy -= input[j3 + 2];
        
            p = resx*resx + resy*resy;
			/* If the resulting value is greater than 255, clip it *
			 * to 255.
			*/
			if (p > 65025)
				output[j2 +1] = 255;      
			else
				output[j2 + 1] = (unsigned char)sqr[p];
			//////////////////////////////////////////////////////////////////
			
			t = (output[j2] - golden[j2])*(output[j2]- golden[j2]);
			PSNR += t;
			t = (output[j2+1] - golden[j2+1])*(output[j2+1]- golden[j2+1]);
			PSNR += t;
			
			
			
		}
		
		for(k = 0; k < t5 ; k++, j++ ) {
			j1 = t1 + j;
            j2 = t2 + j;
            j3 = t3 + j;
			//////////////////////////////////////////////////////////////
			resx = 0;
			resy =0;
			/* Apply the sobel filter and calculate the magnitude *
			 * of the derivative.								  */
			resx -= input[j1 -1];
			resy += input[j1 -1];
			resy += input[j1] << 1;
			resy += input[j1 + 1];
			resx += input[j1 + 1];
			resx -= input[j2 + -1] << 1;
			resx += input[j2 + 1] << 1;
			resx -= input[j3 -1];
			resy -= input[j3 -1];
			resy -= input[j3] << 1;
			resx += input[j3 + 1];
			resy -= input[j3 + 1];
								
			p = resx*resx + resy*resy;
			/* If the resulting value is greater than 255, clip it *
			 * to 255.
										   */
			if (p > 65025)
				output[j2] = 255;      
			else
				output[j2] = (unsigned char)sqr[p];
			///////////////////////////////////////////////////////////////
			
			t = (output[j2] - golden[j2])*(output[j2]- golden[j2]);
			PSNR += t;
			
		}
		t1 += SIZE;
        t2 += SIZE;
        t3 += SIZE;
	}


  
	PSNR /= (double)(SIZE*SIZE);
	PSNR = 10*log10(65536/PSNR);

	/* This is the end of the main computation. Take the end time,  *
	 * calculate the duration of the computation and report it. 	*/
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

	printf ("Total time = %10g seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));

  
	/* Write the output file */
	fwrite(output, sizeof(unsigned char), SIZE*SIZE, f_out);
	fclose(f_out);
  
	return PSNR;
}


int main(int argc, char* argv[])
{
	double PSNR;
	PSNR = sobel(input, output, golden);
	printf("PSNR of original Sobel and computed Sobel image: %g\n", PSNR);
	printf("A visualization of the sobel filter can be found at " OUTPUT_FILE ", or you can run 'make image' to get the jpg\n");

	return 0;
}
