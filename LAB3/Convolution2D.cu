/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	100

 

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;
      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }     

        h_Dst[y * imageW + x] = sum;
      }
    }
  }
        
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
 
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
    
}

__global__ void kernelRow(float *d_Input,float *d_Filter,float *d_OutputGPU,int imageW,int imageH,int filterR)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float sum = 0;

    for(int k = -filterR; k <= filterR; k++) {
	int d = tx + k;
	if( d >= 0 && d < imageW ) {
	    sum += d_Input[ty * imageW + d] * d_Filter[filterR - k];
	}
    }
    d_OutputGPU[ty *imageW + tx] = sum;
}

__global__ void kernelColumn(float *d_Input,float *d_Filter,float *d_OutputGPU,int imageW,int imageH,int filterR)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float sum = 0;

    for(int k = -filterR; k <= filterR; k++) {
	int d = ty + k;
	if( d >= 0 && d < imageH ) {
	    sum += d_Input[d * imageW + tx] * d_Filter[filterR-k];
        }
    }
    d_OutputGPU[ty * imageW + tx] = sum;
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    float
    *h_Filter=0,
    *h_Input=0,
    *h_Buffer=0,
    *h_OutputCPU=0,
    *d_Filter=0,
    *d_Input=0,
    *d_Buffer=0,
    *d_OutputGPU=0;

    int imageW;
    int imageH;
    unsigned int i;
    
    cudaError_t error;

    dim3 grid = 1, block;    

    printf("Enter filter radius : ");
    if( scanf("%d", &filter_radius) == EOF ) {
	printf("Error reading\n");
	return(1);
    }

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    if( scanf("%d", &imageW) == EOF ) {
	printf("Error reading\n");
	return(1);
    }
    imageH = imageW;

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
   
    if( h_Filter==0 || h_Input==0 || h_Buffer==0 || h_OutputCPU==0 ) {
	printf("Could not allocate memory\n");
	if(h_Filter == 0){
	    free(h_Filter);
	}
	if(h_Input == 0){
	    free(h_Input);
	}
	if(h_Buffer == 0){
	    free(h_Buffer);
	}
	if(h_OutputCPU == 0){
	    free(h_OutputCPU);
	}
	return(1);
    }

    if( cudaMalloc((void**)&d_Filter, FILTER_LENGTH * sizeof(float)) != cudaSuccess ) {
	printf("Could not allocate memory\n");
	free(h_Filter);
	free(h_Input);
	free(h_Buffer);
	free(h_OutputCPU);
	cudaDeviceReset();
	return(1);
    }
    if( cudaMalloc((void**)&d_Input, imageW * imageH * sizeof(float)) != cudaSuccess ) {
	printf("Could not allocate memory\n");
	free(h_Filter);
	free(h_Input);
	free(h_Buffer);
	free(h_OutputCPU);
	cudaFree(d_Filter);
	cudaDeviceReset();
	return(1);
    }
    if( cudaMalloc((void**)&d_Buffer, imageW * imageH * sizeof(float)) != cudaSuccess ) {
	printf("Could not allocate memory\n");
	free(h_Filter);
	free(h_Input);
	free(h_Buffer);
	free(h_OutputCPU);
	cudaFree(d_Filter);
	cudaFree(d_Input);
	cudaDeviceReset();
	return(1);
    }
    if( cudaMalloc((void**)&d_OutputGPU, imageW * imageH * sizeof(float)) != cudaSuccess ) {
	printf("Could not allocate memory\n");
	free(h_Filter);
	free(h_Input);
	free(h_Buffer);
	free(h_OutputCPU);
	cudaFree(d_Filter);
	cudaFree(d_Input);
	cudaFree(d_Buffer);
	cudaDeviceReset();
	return(1);
    }
    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
    }
    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles

    //memory copy form host to device
    if( cudaMemcpy(d_Filter,h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ) {
	printf("Problem in memeory copy\n");
	cudaFree(d_Filter);
	cudaFree(d_Input);
	cudaFree(d_Buffer);
	cudaFree(d_OutputGPU);
	cudaDeviceReset();
	free(h_Filter);
	free(h_Buffer);
	free(h_OutputCPU);
	free(h_Input);
	return(1);
    }
    if( cudaMemcpy(d_Input,h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ) {
	printf("Problem in memeory copy\n");
	cudaFree(d_Filter);
	cudaFree(d_Input);
	cudaFree(d_Buffer);
	cudaFree(d_OutputGPU);
	cudaDeviceReset();
        free(h_Filter);
	free(h_Buffer);
	free(h_OutputCPU);
	free(h_Input);
	return(1);
    }
    

        
    //The code below is executed in GPU     
    printf("GPU computation...\n");

    block.x = imageW;
    block.y = imageH;

    kernelRow<<<grid,block>>>(d_Input,d_Filter,d_Buffer,imageW,imageH,filter_radius);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if( error != cudaSuccess ) {
	printf("Cuda Error 1: %s\n", cudaGetErrorString(error));
	cudaFree(d_Filter);
	cudaFree(d_Input);
	cudaFree(d_Buffer);
	cudaFree(d_OutputGPU);
	cudaDeviceReset();
        free(h_Filter);
	free(h_Buffer);
	free(h_OutputCPU);
	free(h_Input);
	return(1);
    }
    kernelColumn<<<grid,block>>>(d_Buffer,d_Filter,d_OutputGPU,imageW,imageH,filter_radius);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if( error != cudaSuccess ) {
	printf("Cuda Error 2: %s\n", cudaGetErrorString(error));
	cudaFree(d_Filter);
	cudaFree(d_Input);
	cudaFree(d_Buffer);
	cudaFree(d_OutputGPU);
	cudaDeviceReset();
        free(h_Filter);
	free(h_Buffer);
	free(h_OutputCPU);
	free(h_Input);
	return(1);
    }
    if( cudaMemcpy(h_Buffer,d_OutputGPU, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ) {
	printf("Problem in memeory copy\n");
	cudaFree(d_Filter);
	cudaFree(d_Input);
	cudaFree(d_Buffer);
	cudaFree(d_OutputGPU);
	cudaDeviceReset();
        free(h_Filter);
	free(h_Buffer);
	free(h_OutputCPU);
	free(h_Input);
	return(1);
    }
    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  

    for(i = 0; i < imageW * imageH; i++ ) {
	if( ABS(h_Buffer[i]-h_OutputCPU[i]) > accuracy ) {
	    printf("Matching failure in %d GPU: %f, CPU: %f\n",i, h_Buffer[i], h_OutputCPU[i]);
	    cudaFree(d_Filter);
	    cudaFree(d_Input);
	    cudaFree(d_Buffer);
	    cudaFree(d_OutputGPU);
	    cudaDeviceReset();
            free(h_Filter);
	    free(h_Buffer);
	    free(h_OutputCPU);
	    free(h_Input);
	    return(1);
	}
    }
    printf("Check completed.\n");	

    // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cudaFree(d_Filter);
    cudaFree(d_Input);
    cudaFree(d_Buffer);
    cudaFree(d_OutputGPU);
    cudaDeviceReset();   

    return 0;
}
