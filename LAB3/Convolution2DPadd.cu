/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	100
#define MyType		float
								 

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(MyType *h_Dst, MyType *h_Src, MyType *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int k, i, j;
  
  for (i =(imageW+2*filter_radius)*filter_radius+filter_radius, j = 1; i < (imageW+2*filter_radius)*(imageH+filter_radius);i++, j++) {
    MyType sum = 0;
    //printf("(Rowcpu)i= %d\n",i);
    for( k = -filterR; k <= filterR; k++) {
      sum += h_Src[i + k] * h_Filter[filterR - k];
      //printf("i+k=%d, filterR-k=%d\n",i+k,filterR-k);
    }
    h_Dst[i] = sum;
    //printf("sum= %f\n",sum);
    if(j == imageW) {
      i=i+(2*filter_radius);
	j=0;
    }
  }     
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(MyType *h_Dst, MyType *h_Src, MyType *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int k, i, j;
  
  for (i =(imageW+2*filter_radius)*filter_radius+filter_radius, j = 1; i < (imageW+2*filter_radius)*(imageH+filter_radius);i++, j++) {
    MyType sum = 0;
    //printf("(Colcpu)i= %d\n",i);
    for(k = -filterR; k <= filterR; k++) {
      sum += h_Src[i+k*(imageW+2*filterR)] * h_Filter[filterR-k];
      //printf("i+k*(imageW+2*filterR)=%d, filterR-k=%d\n",i+k*(imageW+2*filterR),filterR-k);
    }
    h_Dst[i] = sum;
    //printf("sum= %f\n",sum);
    if(j == imageW) {
      i=i+(2*filter_radius);
	j=0;
    }
  }
}

__global__ void kernelRow(MyType *d_Input,MyType *d_Filter,MyType *d_OutputGPU,int imageW,int imageH,int filterR)
{
    int tx = blockIdx.x*blockDim.x+threadIdx.x + filterR;
    int ty = blockIdx.y*blockDim.y+threadIdx.y + filterR;
    MyType sum = 0;

    for(int k = -filterR; k <= filterR; k++) {
	sum += d_Input[ty * (imageW+2*filterR) + tx + k] * d_Filter[filterR - k];
    }
    d_OutputGPU[ty * (imageW+2*filterR) + tx] = sum;
}

__global__ void kernelColumn(MyType *d_Input,MyType *d_Filter,MyType *d_OutputGPU,int imageW,int imageH,int filterR)
{
    int tx = blockIdx.x*blockDim.x+threadIdx.x + filterR;
    int ty = blockIdx.y*blockDim.y+threadIdx.y + filterR;
    MyType sum = 0;

    for(int k = -filterR; k <= filterR; k++) {
	sum += d_Input[(ty+k) * (imageW+2*filterR) + tx] * d_Filter[filterR-k];
    }
    d_OutputGPU[ty * (imageW+2*filterR) + tx] = sum;
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    MyType
    *h_Filter=0,
    *h_Input=0,
    *h_Buffer=0,
    *h_OutputCPU=0,
    *d_Filter=0,
    *d_Input=0,
    *d_Buffer=0,
    *d_OutputGPU=0;
	
    MyType MaxDiff = 0.0, element1, element2;

    int imageW;
    int imageH;
    unsigned int i,j;
    int position;
    
    cudaError_t error;
    struct cudaDeviceProp prop;
    int device;
    double timing, cpu_time, gpu_time;
    struct timeval etstart;
    struct timezone tzp;
    dim3 grid, block;    

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
    h_Filter    = (MyType *)malloc(FILTER_LENGTH * sizeof(MyType));
    h_Input     = (MyType *)calloc((imageW+2*filter_radius) * (imageH+2*filter_radius), sizeof(MyType));
    h_Buffer    = (MyType *)calloc((imageW+2*filter_radius) * (imageH+2*filter_radius), sizeof(MyType));
    h_OutputCPU = (MyType *)calloc((imageW+2*filter_radius) * (imageH+2*filter_radius), sizeof(MyType));
   
    if( h_Filter==0 || h_Input==0 || h_Buffer==0 || h_OutputCPU==0 ) {
	printf("Could not allocate memory(CPU)\n");
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

    if( cudaMalloc((void**)&d_Filter, FILTER_LENGTH * sizeof(MyType)) != cudaSuccess ) {
	printf("Could not allocate memory(GPU)\n");
	free(h_Filter);
	free(h_Input);
	free(h_Buffer);
	free(h_OutputCPU);
	cudaDeviceReset();
	return(1);
    }
    if( cudaMalloc((void**)&d_Input, (imageW+2*filter_radius) * (imageH+2*filter_radius) * sizeof(MyType)) != cudaSuccess ) {
	printf("Could not allocate memory(GPU)\n");
	free(h_Filter);
	free(h_Input);
	free(h_Buffer);
	free(h_OutputCPU);
	cudaFree(d_Filter);
	cudaDeviceReset();
	return(1);
    }
    /*Memset is asynchronous with respect to the host.*/
    cudaMemset((void*)d_Input, 0, (imageW+2*filter_radius) * (imageH+2*filter_radius) * sizeof(MyType));
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if( error != cudaSuccess ) {
	printf("Memset1Error: %s\n",cudaGetErrorString(error));
	free(h_Filter);
	free(h_Input);
	free(h_Buffer);
	free(h_OutputCPU);
	cudaFree(d_Filter);
	cudaFree(d_Input);
	cudaDeviceReset();
	return(1);
    }

    if( cudaMalloc((void**)&d_Buffer, (imageW+2*filter_radius) * (imageH+2*filter_radius) * sizeof(MyType)) != cudaSuccess ) {
	printf("Could not allocate memory(GPU)\n");
	free(h_Filter);
	free(h_Input);
	free(h_Buffer);
	free(h_OutputCPU);
	cudaFree(d_Filter);
	cudaFree(d_Input);
	cudaDeviceReset();
	return(1);
    }
    /*Memset is asynchronous with respect to the host.*/
    cudaMemset((void*)d_Buffer, 0, (imageW+2*filter_radius) * (imageH+2*filter_radius) * sizeof(MyType));
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if( error != cudaSuccess ) {
	printf("Memset2Error: %s\n",cudaGetErrorString(error));
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
    if( cudaMalloc((void**)&d_OutputGPU, (imageW+2*filter_radius) * (imageH+2*filter_radius) * sizeof(MyType)) != cudaSuccess ) {
	printf("Could not allocate memory(GPU)\n");
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
    /*Memset is asynchronous with respect to the host.*/
    cudaMemset((void*)d_OutputGPU, 0, (imageW+2*filter_radius) * (imageH+2*filter_radius) * sizeof(MyType));
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if( error != cudaSuccess ) {
	printf("Memset3Error: %s\n",cudaGetErrorString(error));
	free(h_Filter);
	free(h_Input);
	free(h_Buffer);
	free(h_OutputCPU);
	cudaFree(d_Filter);
	cudaFree(d_Input);
	cudaFree(d_Buffer);
        cudaFree(d_OutputGPU);
	cudaDeviceReset();
	return(1);
    }
    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (MyType)(rand() % 16);
    }
    //printf("Init\n");
    for (i =(imageW+2*filter_radius)*filter_radius+filter_radius, j = 1; i < ((imageW+2*filter_radius)*(imageH+filter_radius));i++, j++) {
        h_Input[i] += (MyType)rand() / ((MyType)RAND_MAX / 255) + (MyType)rand() / (MyType)RAND_MAX;
	//printf("i= %d, val = %lf\n", i, h_Input[i]);
	if(j == imageW) {
	    i=i+(2*filter_radius);
	    j=0;
	}
    }

    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");
    if(gettimeofday(&etstart, &tzp) == -1){
        perror("Error calling gettimeofday()\n");
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
    cpu_time = ((double)etstart.tv_sec) + ((double)etstart.tv_usec) / 1000000.0;
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles

    if(gettimeofday(&etstart, &tzp) == -1){
        perror("Error calling gettimeofday()\n");
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

    timing = ((double)etstart.tv_sec) + ((double)etstart.tv_usec) / 1000000.0;
    cpu_time = timing - cpu_time;
    printf("CPU_time is: %lf seconds\n", cpu_time);
    //for(i=0; i < (imageW+2*filter_radius)*(imageH+2*filter_radius);i++){
	//printf("i= %d, val is %lf\n",i,h_Input[i]);
    //}
    cudaGetDevice(&device);
    if( cudaGetDeviceProperties(&prop,device) != cudaSuccess ) {
	printf("Invalid device.\n");
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
    if( (imageW*imageH) > prop.maxThreadsPerBlock ) {
	block.x = sqrt(prop.maxThreadsPerBlock);
	block.y = sqrt(prop.maxThreadsPerBlock);
	grid.x = imageW/block.x;
	grid.y = imageH/block.y;
    }
    else {
	grid = 1;
    	block.x = imageW;
    	block.y = imageH;
    }
    //The code below is executed in GPU     
    printf("GPU computation...\n");
    if(gettimeofday(&etstart, &tzp) == -1){
        perror("Error calling gettimeofday()\n");
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
    gpu_time = ((double)etstart.tv_sec) + ((double)etstart.tv_usec) / 1000000.0;

    //memory copy from host to device
    if( cudaMemcpy(d_Filter,h_Filter, FILTER_LENGTH * sizeof(MyType), cudaMemcpyHostToDevice) != cudaSuccess ) {
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
    if( cudaMemcpy(d_Input,h_Input, (imageW+2*filter_radius) * (imageH+2*filter_radius) * sizeof(MyType), cudaMemcpyHostToDevice) != cudaSuccess ) {
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
    if( cudaMemcpy(h_Buffer,d_OutputGPU, (imageW+2*filter_radius) * (imageH+2*filter_radius) * sizeof(MyType), cudaMemcpyDeviceToHost) != cudaSuccess ) {
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

    if(gettimeofday(&etstart, &tzp) == -1){
        perror("Error calling gettimeofday()\n");
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

    timing = ((double)etstart.tv_sec) + ((double)etstart.tv_usec) / 1000000.0;
    gpu_time = timing - gpu_time;
    printf("GPU_time is: %lf seconds\n", gpu_time);
    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  

    for (i =(imageW+2*filter_radius)*filter_radius+filter_radius, j = 1; i < (imageW+2*filter_radius)*(imageH+filter_radius);i++, j++) {
	if( ABS(h_Buffer[i]-h_OutputCPU[i]) > (MyType)MaxDiff ) {
	    MaxDiff = ABS(h_Buffer[i]-h_OutputCPU[i]);
	    element1 = h_Buffer[i];
	    element2 = h_OutputCPU[i];
	    position = i;
	}
	if(j == imageW) {
	    i=i+(2*filter_radius);
	    j=0;
	}
    }
    printf("Max difference in pos:%d-> %f ->GPU: %f, CPU: %f\n", position, MaxDiff, element1, element2);
    for (i =(imageW+2*filter_radius)*filter_radius+filter_radius, j = 1; i < (imageW+2*filter_radius)*(imageH+filter_radius); i++,j++) {
	if( ABS(h_Buffer[i]-h_OutputCPU[i]) > (MyType)accuracy ) {
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
	if(j == imageW) {
	    i=i+(2*filter_radius);
	    j=0;
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
