# PCA-EXP-3-Simple-Warp-Divergence-Implement-Sum-Reduction
Refer to the kernel reduceUnrolling8 and implement the kernel reduceUnrolling16, in which each thread handles 16 data blocks. Compare kernel performance with reduceUnrolling8 and use the proper metrics and events with nvprof to explain any difference in performance.

## Aim:
Compare the performance of the kernel "reduceUnrolling8" and the newly implemented kernel "reduceUnrolling16" by handling 8 and 16 data blocks per thread respectively.

## PROCEDURE FOR UNROLLIN8:
1. Initialize the input size n and allocate host memory (h_idata and h_odata) for input and 
    output data.

2. Initialize the input data on the host by assigning a value of 1 to each element in h_idata.

3. Allocate device memory (d_idata and d_odata) for input and output data on the GPU.

4. Copy the input data from the host to the device using cudaMemcpy.

5. Define the grid and block dimensions for the kernel launch. Each block will contain 256                        threads, and the grid size will be calculated based on the input size n and block size.

6. Start the CPU timer to measure the CPU execution time.

7. Compute the sum of input data on the CPU using a for loop and store the result in         -----sum_cpu.

8. Stop the CPU timer and calculate the elapsed CPU time.

9. Start the GPU timer to measure the GPU execution time.

10. Launch the reduceUnrolling8 kernel on the GPU with the specified grid and block dimensions.

11. Copy the result data from the device to the host using cudaMemcpy.

12. Compute the final sum on the GPU by adding up the elements in h_odata and store the result in sum_gpu.

13. Stop the GPU timer and calculate the elapsed GPU time.

14. Print the results: CPU sum, GPU sum, CPU elapsed time, and GPU elapsed time.

15. Free the allocated host and device memory using free and cudaFree.

16. Return from the main function.

## Program:
reduceInteger.cu:
## U8.cu:
```
#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n)
{
    // Set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // Convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // Unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // In-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // Synchronize within threadblock
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

// Function to calculate elapsed time in milliseconds
double getElapsedTime(struct timeval start, struct timeval end)
{
    long seconds = end.tv_sec - start.tv_sec;
    long microseconds = end.tv_usec - start.tv_usec;
    double elapsed = seconds + microseconds / 1e6;
    return elapsed * 1000; // Convert to milliseconds
}

int main()
{
    // Input size and host memory allocation
    unsigned int n = 1 << 20; // 1 million elements
    size_t size = n * sizeof(int);
    int *h_idata = (int *)malloc(size);
    int *h_odata = (int *)malloc(size);

    // Initialize input data on the host
    for (unsigned int i = 0; i < n; i++)
    {
        h_idata[i] = 1;
    }

    // Device memory allocation
    int *d_idata, *d_odata;
    cudaMalloc((void **)&d_idata, size);
    cudaMalloc((void **)&d_odata, size);

    // Copy input data from host to device
    cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(256); // 256 threads per block
    dim3 gridSize((n + blockSize.x * 8 - 1) / (blockSize.x * 8));

    // Start CPU timer
    struct timeval start_cpu, end_cpu;
    gettimeofday(&start_cpu, NULL);

    // Compute the sum on the CPU
    int sum_cpu = 0;
    for (unsigned int i = 0; i < n; i++)
    {
        sum_cpu += h_idata[i];
    }

    // Stop CPU timer
    gettimeofday(&end_cpu, NULL);
    double elapsedTime_cpu = getElapsedTime(start_cpu, end_cpu);

    // Start GPU timer
    struct timeval start_gpu, end_gpu;
    gettimeofday(&start_gpu, NULL);

    // Launch the reduction kernel
    reduceUnrolling8<<<gridSize, blockSize>>>(d_idata, d_odata, n);

    // Copy the result from device to host
    cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost);

    // Compute the final sum on the GPU
    int sum_gpu = 0;
    for (unsigned int i = 0; i < gridSize.x; i++)
    {
        sum_gpu += h_odata[i];
    }

    // Stop GPU timer
    gettimeofday(&end_gpu, NULL);
    double elapsedTime_gpu = getElapsedTime(start_gpu, end_gpu);

    // Print the results and elapsed times
    printf("CPU Sum: %d\n", sum_cpu);
    printf("GPU Sum: %d\n", sum_gpu);
    printf("CPU Elapsed Time: %.2f ms\n", elapsedTime_cpu);
    printf("GPU Elapsed Time: %.2f ms\n", elapsedTime_gpu);

    // Free memory
    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);

    return 0;
}
```
![image](https://github.com/Siddarthan999/PCA-EXP-3-Simple-Warp-Divergence-Implement-Sum-Reduction/assets/91734840/6b1a9c99-5a69-41d2-8eda-d9c12a99ce93)

## PROCEDURE FOR UNROLLING16:
1. Initialize the input size n and allocate host memory (h_idata and h_odata) for input and 
    output data.

2. Initialize the input data on the host by assigning a value of 1 to each element in h_idata.

3. Allocate device memory (d_idata and d_odata) for input and output data on the GPU.

4. Copy the input data from the host to the device using cudaMemcpy.

5. Define the grid and block dimensions for the kernel launch. Each block will contain 256                        threads, and the grid size will be calculated based on the input size n and block size.

6. Start the CPU timer to measure the CPU execution time.

7. Compute the sum of input data on the CPU using a for loop and store the result in         -----sum_cpu.

8. Stop the CPU timer and calculate the elapsed CPU time.

9. Start the GPU timer to measure the GPU execution time.

10. Launch the reduceUnrolling16 kernel on the GPU with the specified grid and block dimensions.

11. Copy the result data from the device to the host using cudaMemcpy.

12. Compute the final sum on the GPU by adding up the elements in h_odata and store the result in sum_gpu.

13. Stop the GPU timer and calculate the elapsed GPU time.

14. Print the results: CPU sum, GPU sum, CPU elapsed time, and GPU elapsed time.

15. Free the allocated host and device memory using free and cudaFree.

16. Return from the main function.

## U16.cu:
```
#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
// Kernel function declaration
__global__ void reduceUnrolling16(int *g_idata, int *g_odata, unsigned int n);
// Function to calculate elapsed time in milliseconds
double getElapsedTime(struct timeval start, struct timeval end)
{
    long seconds = end.tv_sec - start.tv_sec;
    long microseconds = end.tv_usec - start.tv_usec;
    double elapsed = seconds + microseconds / 1e6;
    return elapsed * 1000; // Convert to milliseconds
}
int main()
{
    // Input size and host memory allocation
    unsigned int n = 1 << 20; // 1 million elements
    size_t size = n * sizeof(int);
    int *h_idata = (int *)malloc(size);
    int *h_odata = (int *)malloc(size);

    // Initialize input data on the host
    for (unsigned int i = 0; i < n; i++)
    {
        h_idata[i] = 1;
    }

    // Device memory allocation
    int *d_idata, *d_odata;
    cudaMalloc((void **)&d_idata, size);
    cudaMalloc((void **)&d_odata, size);

    // Copy input data from host to device
    cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(256); // 256 threads per block
    dim3 gridSize((n + blockSize.x * 16 - 1) / (blockSize.x * 16));

    // Start CPU timer
    struct timeval start_cpu, end_cpu;
    gettimeofday(&start_cpu, NULL);

    // Compute the sum on the CPU
    int sum_cpu = 0;
    for (unsigned int i = 0; i < n; i++)
    {
        sum_cpu += h_idata[i];
    }

    // Stop CPU timer
    gettimeofday(&end_cpu, NULL);
    double elapsedTime_cpu = getElapsedTime(start_cpu, end_cpu);

    // Start GPU timer
    struct timeval start_gpu, end_gpu;
    gettimeofday(&start_gpu, NULL);

    // Launch the reduction kernel
    reduceUnrolling16<<<gridSize, blockSize>>>(d_idata, d_odata, n);

    // Copy the result from device to host
    cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost);

    // Compute the final sum on the GPU
    int sum_gpu = 0;
    for (unsigned int i = 0; i < gridSize.x; i++)
    {
        sum_gpu += h_odata[i];
    }

    // Stop GPU timer
    gettimeofday(&end_gpu, NULL);
    double elapsedTime_gpu = getElapsedTime(start_gpu, end_gpu);

    // Print the results and elapsed times
    printf("CPU Sum: %d\n", sum_cpu);
    printf("GPU Sum: %d\n", sum_gpu);
    printf("CPU Elapsed Time: %.2f ms\n", elapsedTime_cpu);
    printf("GPU Elapsed Time: %.2f ms\n", elapsedTime_gpu);

    // Free memory
    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);

    return 0;
}

__global__ void reduceUnrolling16(int *g_idata, int *g_odata, unsigned int n)
{
    // Set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 16 + threadIdx.x;

    // Convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 16;

    // Unrolling 16
    if (idx + 15 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int a5 = g_idata[idx + 4 * blockDim.x];
        int a6 = g_idata[idx + 5 * blockDim.x];
        int a7 = g_idata[idx + 6 * blockDim.x];
        int a8 = g_idata[idx + 7 * blockDim.x];
        int b1 = g_idata[idx + 8 * blockDim.x];
        int b2 = g_idata[idx + 9 * blockDim.x];
        int b3 = g_idata[idx + 10 * blockDim.x];
        int b4 = g_idata[idx + 11 * blockDim.x];
        int b5 = g_idata[idx + 12 * blockDim.x];
        int b6 = g_idata[idx + 13 * blockDim.x];
        int b7 = g_idata[idx + 14 * blockDim.x];
        int b8 = g_idata[idx + 15 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8;
    }

    __syncthreads();

    // In-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // Synchronize within thread block
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}
```
![image](https://github.com/Siddarthan999/PCA-EXP-3-Simple-Warp-Divergence-Implement-Sum-Reduction/assets/91734840/5693fa91-10a2-48c3-a0ae-0bf52840aa7a)

## Result:
Thus, the program has been executed by unrolling by 8 and unrolling by 16. It is observed that Unrolling by 8 has executed with less elapsed time than unrolling by 16 with blocks 16.
