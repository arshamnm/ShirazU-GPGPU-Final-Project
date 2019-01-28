#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#include "utils.h"

__global__
void histogram_kernel(unsigned int* d_bins, const float* d_in, const int bin_count, const float lum_min, const float lum_max, const int size) {  
    int mid = threadIdx.x + blockDim.x * blockIdx.x;
    if(mid >= size)
        return;
    float lum_range = lum_max - lum_min;
    int bin = ((d_in[mid]-lum_min) / lum_range) * bin_count;
    
    atomicAdd(&d_bins[bin], 1);
}

__global__ 
void scan_kernel(unsigned int* d_bins, int size) {
    int mid = threadIdx.x + blockDim.x * blockIdx.x;
    if(mid >= size)
        return;
    
    for(int s = 1; s <= size; s *= 2) {
          int spot = mid - s; 
         
          unsigned int val = 0;
          if(spot >= 0)
              val = d_bins[spot];
          __syncthreads();
          if(spot >= 0)
              d_bins[mid] += val;
          __syncthreads();

    }
}
// calculate reduce max or min and stick the value in d_answer.
__global__
void reduce_minmax_kernel(const float* const d_in, float* d_out, const size_t size, int minmax) {
    extern __shared__ float shared[];
    
    int mid = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x; 
    
    // we have 1 thread per block, so copying the entire block should work fine
    if(mid < size) {
        shared[tid] = d_in[mid];
    } else {
        if(minmax == 0)
            shared[tid] = FLT_MAX;
        else
            shared[tid] = -FLT_MAX;
    }
    
    // wait for all threads to copy the memory
    __syncthreads();
    
    // don't do any thing with memory if we happen to be far off ( I don't know how this works with
    // sync threads so I moved it after that point )
    if(mid >= size) {   
        if(tid == 0) {
            if(minmax == 0) 
                d_out[blockIdx.x] = FLT_MAX;
            else
                d_out[blockIdx.x] = -FLT_MAX;

        }
        return;
    }
       
    for(unsigned int s = blockDim.x/2; s > 0; s /= 2) {
        if(tid < s) {
            if(minmax == 0) {
                shared[tid] = min(shared[tid], shared[tid+s]);
            } else {
                shared[tid] = max(shared[tid], shared[tid+s]);
            }
        }
        
        __syncthreads();
    }
    
    if(tid == 0) {
        d_out[blockIdx.x] = shared[0];
    }
}

int get_max_size(int n, int d) {
    return (int)ceil( (float)n/(float)d ) + 1;
}

float reduce_minmax(const float* const d_in, const size_t size, int minmax) {
    int BLOCK_SIZE = 1024;
    // we need to keep reducing until we get to the amount that we consider 
    // having the entire thing fit into one block size
    size_t curr_size = size;
    float* d_curr_in;
    
    checkCudaErrors(cudaMalloc(&d_curr_in, sizeof(float) * size));    
    checkCudaErrors(cudaMemcpy(d_curr_in, d_in, sizeof(float) * size, cudaMemcpyDeviceToDevice));


    float* d_curr_out;
    
    dim3 thread_dim(BLOCK_SIZE);
    const int shared_mem_size = sizeof(float)*BLOCK_SIZE;
    
    while(1) {
        checkCudaErrors(cudaMalloc(&d_curr_out, sizeof(float) * get_max_size(curr_size, BLOCK_SIZE)));
        
        dim3 block_dim(get_max_size(size, BLOCK_SIZE));
        reduce_minmax_kernel<<<block_dim, thread_dim, shared_mem_size>>>(
            d_curr_in,
            d_curr_out,
            curr_size,
            minmax
        );
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

            
        // move the current input to the output, and clear the last input if necessary
        checkCudaErrors(cudaFree(d_curr_in));
        d_curr_in = d_curr_out;
        
        if(curr_size <  BLOCK_SIZE) 
            break;
        
        curr_size = get_max_size(curr_size, BLOCK_SIZE);
    }
    
    // theoretically we should be 
    float h_out;
    cudaMemcpy(&h_out, d_curr_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_curr_out);
    return h_out;
}

void histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
    const size_t size = numRows*numCols;
    min_logLum = reduce_minmax(d_logLuminance, size, 0);
    max_logLum = reduce_minmax(d_logLuminance, size, 1);
    
    printf("got min of %f\n", min_logLum);
    printf("got max of %f\n", max_logLum);
    printf("numBins %d\n", numBins);
    
    unsigned int* d_bins;
    size_t histo_size = sizeof(unsigned int)*numBins;

    checkCudaErrors(cudaMalloc(&d_bins, histo_size));    
    checkCudaErrors(cudaMemset(d_bins, 0, histo_size));  
    dim3 thread_dim(1024);
    dim3 hist_block_dim(get_max_size(size, thread_dim.x));
    histogram_kernel<<<hist_block_dim, thread_dim>>>(d_bins, d_logLuminance, numBins, min_logLum, max_logLum, size);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    dim3 scan_block_dim(get_max_size(numBins, thread_dim.x));

    scan_kernel<<<scan_block_dim, thread_dim>>>(d_bins, numBins); // converts histogram to cdf
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    cudaMemcpy(d_cdf, d_bins, histo_size, cudaMemcpyDeviceToDevice);

    
    checkCudaErrors(cudaFree(d_bins));
}
