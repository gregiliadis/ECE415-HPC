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
#define MyType		double
								 
__constant__ MyType filter[161];
////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(MyType *h_Dst, MyType *h_Src, MyType *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int k, i, j;
  
  for (i =(imageW+2*filter_radius)*filter_radius+filter_radius, j = 1; i < (imageW+2*filter_radius)*(imageH+filter_radius);i++, j++) {
    MyType sum = 0;
    for( k = -filterR; k <= filterR; k++) {
      sum += h_Src[i + k] * h_Filter[filterR - k];
    }
    h_Dst[i] = sum;
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
    for(k = -filterR; k <= filterR; k++) {
      sum += h_Src[i+k*(imageW+2*filterR)] * h_Filter[filterR-k];
    }
    h_Dst[i] = sum;
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
    int input_pos = ty * (imageW+2*filterR) + tx;
    int row_length = 2*filterR + blockDim.x;
    int thread_row = threadIdx.y*row_length;
    
    
    //Array's size will be dynamically allocated at kernel launch time
    extern __shared__ MyType s_Input[];
    
    for(int k = 0; k + threadIdx.x < row_length ; k+= blockDim.x){
	s_Input[thread_row + threadIdx.x + k] = d_Input[input_pos - filterR + k];
    }

    __syncthreads();


    for(int k = -filterR; k <= filterR; k++) {
	sum += s_Input[thread_row + threadIdx.x + filterR + k] * filter[filterR - k];
    }
    d_OutputGPU[input_pos] = sum;
}

__global__ void kernelColumn(MyType *d_Input,MyType *d_Filter,MyType *d_OutputGPU,int imageW,int imageH,int filterR)
{
    int tx = blockIdx.x*blockDim.x+threadIdx.x + filterR;
    int ty = blockIdx.y*blockDim.y+threadIdx.y + filterR;
    MyType sum = 0;
    int input_pos = ty * (imageW+2*filterR) + tx;
    int col_length = blockDim.y + 2*filterR;
    int thread_row = threadIdx.y * blockDim.x;
    int input_row_len = imageW+2*filterR;
    
    //Array's size will be dynamically allocated at kernel launch time
    extern __shared__ MyType s_Input[];

    for(int k = 0; k + threadIdx.y < col_length; k+= blockDim.y ){
	s_Input[thread_row + k*blockDim.x + threadIdx.x] = d_Input[input_pos + (k - filterR)*input_row_len];
    }

    __syncthreads();

    for(int k = -filterR; k <= filterR; k++) {
	sum += s_Input[thread_row + (filterR + k)*blockDim.x + threadIdx.x] * filter[filterR-k];
    }
    d_OutputGPU[input_pos] = sum;
}

void free_all(void *d_Input,void *d_OutputGPU,void *h_Filter,void *h_Buffer,void *h_OutputCPU,void *h_Input) { 
	cudaFree(d_Input);
	cudaFree(d_OutputGPU);
	cudaDeviceReset();
        free(h_Filter);
	cudaFreeHost(h_Buffer);
	cudaFreeHost(h_OutputCPU);
	cudaFreeHost(h_Input);
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
    *d_Input=0,
    *d_OutputGPU=0;
	
    MyType MaxDiff = 0.0, element1, element2;

    size_t imageW, imageH;
    unsigned int i,j;
    int position;
    int tile_size, num_of_tiles, tile_pos, tile_height;
    
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
	
    printf("Enter tile size : ");
    if( scanf("%d", &tile_size) == EOF ) {
	printf("Error reading\n");
	return(1);
    }
 
    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    if( scanf("%zd", &imageW) == EOF ) {
	printf("Error reading\n");
	return(1);
    }
    imageH = imageW;

    printf("Image Width x Height = %lu x %lu\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    
    h_Filter = (MyType *)malloc(FILTER_LENGTH * sizeof(MyType));  
    if( h_Filter==0 ) {
	printf("Could not allocate memory(CPU)\n");
	return(1);
    }

    if( cudaHostAlloc((void**)&h_Input,(size_t)((imageW+2*filter_radius)*(imageH+2*filter_radius)*(size_t)sizeof(MyType)),cudaHostAllocDefault) != cudaSuccess){
	    printf("cudaHostAlloc error in h_Input.\n");
	    return(1);
    }
    memset((void*)h_Input,0,(size_t)((imageW+2*filter_radius)*(imageH+2*filter_radius)*(size_t)sizeof(MyType)));
    if( cudaHostAlloc((void**)&h_Buffer,(size_t)((imageW+2*filter_radius)*(imageH+2*filter_radius)*(size_t)sizeof(MyType)),cudaHostAllocDefault) != cudaSuccess ) {
	    printf("CudaHostAlloc error in h_Buffer.\n");
	    cudaFreeHost(h_Input);
	    return(1);
    }
    memset((void*)h_Buffer,0,(size_t)((imageW+2*filter_radius)*(imageH+2*filter_radius)*(size_t)sizeof(MyType)));
    if( cudaHostAlloc((void**)&h_OutputCPU,(size_t)((imageW+2*filter_radius)*(imageH+2*filter_radius)*(size_t)sizeof(MyType)),cudaHostAllocDefault) != cudaSuccess ) {
	    printf("CudAhostAlloc error in h_OutputGPU.\n");
	    cudaFreeHost(h_Input);
	    cudaFreeHost(h_Buffer);
	    return(1);
    }
    memset((void*)h_OutputCPU,0,(size_t)((imageW+2*filter_radius)*(imageH+2*filter_radius)*(size_t)sizeof(MyType)));
    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (MyType)(rand() % 16);
    }
    for (i =(imageW+2*filter_radius)*filter_radius+filter_radius, j = 1; i < ((imageW+2*filter_radius)*(imageH+filter_radius));i++, j++) {
        h_Input[i] += (MyType)rand() / ((MyType)RAND_MAX / 255) + (MyType)rand() / (MyType)RAND_MAX;
	if(j == imageW) {
	    i=i+(2*filter_radius);
	    j=0;
	}
    }

    printf("CPU computation...\n");
    if(gettimeofday(&etstart, &tzp) == -1){
        perror("Error calling gettimeofday()\n");
        free(h_Filter);
	cudaFreeHost(h_Buffer);
	cudaFreeHost(h_OutputCPU);
	cudaFreeHost(h_Input);
	return(1);
    }
    cpu_time = ((double)etstart.tv_sec) + ((double)etstart.tv_usec) / 1000000.0;
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles

    if(gettimeofday(&etstart, &tzp) == -1){
        perror("Error calling gettimeofday()\n");
        free(h_Filter);
	cudaFreeHost(h_Buffer);
	cudaFreeHost(h_OutputCPU);
	cudaFreeHost(h_Input);
	return(1);
    }

    timing = ((double)etstart.tv_sec) + ((double)etstart.tv_usec) / 1000000.0;
    cpu_time = timing - cpu_time;
    printf("CPU_time is: %lf seconds\n", cpu_time);
    cudaGetDevice(&device);
    if( cudaGetDeviceProperties(&prop,device) != cudaSuccess ) {
	printf("Invalid device.\n");
        free(h_Filter);
	cudaFreeHost(h_Buffer);
	cudaFreeHost(h_OutputCPU);
	cudaFreeHost(h_Input);
	return(1);
    }

    num_of_tiles = (imageW*imageH)/tile_size;

    if( imageW > sqrt(prop.maxThreadsPerBlock) ) {
    	    block.x = sqrt(prop.maxThreadsPerBlock);
	    grid.x = imageW/block.x;
	    if( tile_size/imageW > sqrt(prop.maxThreadsPerBlock)){
		    block.y = sqrt(prop.maxThreadsPerBlock);
		    grid.y = (tile_size/imageW)/sqrt(prop.maxThreadsPerBlock);
	    }
	    else{
	    	    block.y = tile_size/imageW;
		    grid.y = 1;
	    }
	    tile_height = tile_size/imageW;
    }
    else{
            grid=1;
	    block.x = imageW;
	    block.y = imageH;
	    num_of_tiles=1;
	    tile_height = imageH;
    }
	
    if( cudaMalloc((void**)&d_Input, (imageW+2*filter_radius)*(tile_height+2*filter_radius)*sizeof(MyType)) != cudaSuccess ) {
	printf("Could not allocate memory(GPU)\n");
	free(h_Filter);
	cudaFreeHost(h_Input);
	cudaFreeHost(h_Buffer);
	cudaFreeHost(h_OutputCPU);
	cudaDeviceReset();
	return(1);
    }
    /*Memset is asynchronous with respect to the host.*/
    cudaMemset((void*)d_Input, 0, (imageW+2*filter_radius)*(tile_height+2*filter_radius)*sizeof(MyType));
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if( error != cudaSuccess ) {
	printf("Memset1Error: %s\n",cudaGetErrorString(error));
	free(h_Filter);
	cudaFreeHost(h_Input);
	cudaFreeHost(h_Buffer);
	cudaFreeHost(h_OutputCPU);
	cudaFree(d_Input);
	cudaDeviceReset();
	return(1);
    }	

    if( cudaMalloc((void**)&d_OutputGPU, (imageW+2*filter_radius)*(tile_height+2*filter_radius)*sizeof(MyType)) != cudaSuccess ) {
	printf("Could not allocate memory(GPU)\n");
	free(h_Filter);
	cudaFreeHost(h_Input);
	cudaFreeHost(h_Buffer);
	cudaFreeHost(h_OutputCPU);
	cudaFree(d_Input);
	cudaDeviceReset();
	return(1);
    }
    /*Memset is asynchronous with respect to the host.*/
    cudaMemset((void*)d_OutputGPU, 0, (imageW+2*filter_radius)*(tile_height+2*filter_radius)*sizeof(MyType));
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if( error != cudaSuccess ) {
	printf("Memset3Error: %s\n",cudaGetErrorString(error));
	free(h_Filter);
	cudaFreeHost(h_Input);
	cudaFreeHost(h_Buffer);
	cudaFreeHost(h_OutputCPU);
	cudaFree(d_Input);
        cudaFree(d_OutputGPU);
	cudaDeviceReset();
	return(1);
    }
    
    //The code below is executed in GPU     
    printf("GPU computation...\n");
    if(gettimeofday(&etstart, &tzp) == -1){
        perror("Error calling gettimeofday()\n");
	free_all(d_Input,d_OutputGPU,h_Filter,h_Buffer,h_OutputCPU,h_Input);
	return(1);
    }
    gpu_time = ((double)etstart.tv_sec) + ((double)etstart.tv_usec) / 1000000.0;

    //memory copy from host to device
    if( cudaMemcpyToSymbol(filter,h_Filter, FILTER_LENGTH * sizeof(MyType), 0, cudaMemcpyHostToDevice) != cudaSuccess ) {
	printf("Problem in memory copy\n");
	free_all(d_Input,d_OutputGPU,h_Filter,h_Buffer,h_OutputCPU,h_Input);
	return(1);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    tile_pos = 0;
    for(i=0; i<num_of_tiles;i++){
    	if( cudaMemcpy(d_Input,h_Input+tile_pos, (imageW+2*filter_radius) * (tile_height+2*filter_radius) * sizeof(MyType), cudaMemcpyHostToDevice) != cudaSuccess ) {
		printf("Problem in memory copy\n");
		free_all(d_Input,d_OutputGPU,h_Filter,h_Buffer,h_OutputCPU,h_Input);
		return(1);
    	}
    	kernelRow<<<grid,block, block.y *(block.x+2*filter_radius)*sizeof(MyType)>>>(d_Input,filter,d_OutputGPU,imageW,tile_height,filter_radius);
    	//cudaDeviceSynchronize();
    	error = cudaGetLastError();
    	if( error != cudaSuccess ) {
		printf("Cuda Error 1: %s\n", cudaGetErrorString(error));
		free_all(d_Input,d_OutputGPU,h_Filter,h_Buffer,h_OutputCPU,h_Input);
		return(1);
   	}
    	
	if( cudaMemcpy(h_Buffer+((imageW+2*filter_radius)*filter_radius)+tile_pos,
				d_OutputGPU+((imageW+2*filter_radius)*filter_radius),
			       	(imageW+2*filter_radius) * tile_height * sizeof(MyType), cudaMemcpyDeviceToHost) != cudaSuccess ) {
		printf("Problem in memeory copy\n");
		free_all(d_Input,d_OutputGPU,h_Filter,h_Buffer,h_OutputCPU,h_Input);
		return(1);
    	}
    	tile_pos+= (imageW+2*filter_radius) * tile_height;
    }
    cudaDeviceSynchronize();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    tile_pos = 0;
    for(i=0; i<num_of_tiles;i++){
    	if( cudaMemcpy(d_Input,h_Buffer+tile_pos, (imageW+2*filter_radius) * (tile_height+2*filter_radius) * sizeof(MyType), cudaMemcpyHostToDevice) != cudaSuccess ) {
		printf("Problem in memory copy\n");
		free_all(d_Input,d_OutputGPU,h_Filter,h_Buffer,h_OutputCPU,h_Input);
		return(1);
    	}
    	kernelColumn<<<grid,block, block.x *(block.y+2*filter_radius)*sizeof(MyType)>>>(d_Input,filter,d_OutputGPU,imageW,tile_height,filter_radius);
    	//cudaDeviceSynchronize();
    	error = cudaGetLastError();
    	if( error != cudaSuccess ) {
		printf("Cuda Error 1: %s\n", cudaGetErrorString(error));
		free_all(d_Input,d_OutputGPU,h_Filter,h_Buffer,h_OutputCPU,h_Input);
		return(1);
   	}
    	
	if( cudaMemcpy(h_Input+((imageW+2*filter_radius)*filter_radius)+tile_pos,
				d_OutputGPU+((imageW+2*filter_radius)*filter_radius),
			       	(imageW+2*filter_radius) * tile_height * sizeof(MyType), cudaMemcpyDeviceToHost) != cudaSuccess ) {
		printf("Problem in memeory copy\n");
		free_all(d_Input,d_OutputGPU,h_Filter,h_Buffer,h_OutputCPU,h_Input);
		return(1);
    	}
    	tile_pos+= (imageW+2*filter_radius) * tile_height;
    }

    if(gettimeofday(&etstart, &tzp) == -1){
        perror("Error calling gettimeofday()\n");
        free_all(d_Input, d_OutputGPU,h_Filter, h_Input, h_Buffer, h_OutputCPU);
	return(1);
    }

    timing = ((double)etstart.tv_sec) + ((double)etstart.tv_usec) / 1000000.0;
    gpu_time = timing - gpu_time;
    printf("GPU_time is: %lf seconds\n", gpu_time);
    
    for (i =(imageW+2*filter_radius)*filter_radius+filter_radius, j = 1; i < (imageW+2*filter_radius)*(imageH+filter_radius);i++, j++) {
	if( ABS(h_Input[i]-h_OutputCPU[i]) > (MyType)MaxDiff ) {
	    MaxDiff = ABS(h_Input[i]-h_OutputCPU[i]);
	    element1 = h_Input[i];
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
	if( ABS(h_Input[i]-h_OutputCPU[i]) > (MyType)accuracy ) {
	    printf("Matching failure in %d GPU: %f, CPU: %f\n",i, h_Input[i], h_OutputCPU[i]);
	    free_all(d_Input,d_OutputGPU,h_Filter,h_Buffer,h_Input,h_OutputCPU);
	    return(1);
	}
	if(j == imageW) {
	    i=i+(2*filter_radius);
	    j=0;
	}
    }

    printf("Check completed.\n");	

    //free all host and device allocated memory
    free_all(d_Input,d_OutputGPU,h_Filter,h_Buffer,h_Input,h_OutputCPU);

    return 0;
}
