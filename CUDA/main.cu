#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

float *GPU_fill(float *matrix, int rows, int cols)
{
    int i;
    for (i = 0; i < rows * cols; ++i)
    {
        matrix[i] = rand() % 10;
    }
    return matrix;
}

void print(float *matrix, const int rows, const int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f\t", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void cuda_multiply(float *A, float *B, float *C, const int m, const int n, const int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m)
    {
        C[row * n + col] = 0;

        for (int i = 0; i < k; i++)
        {
            C[row * n + col] += A[row * k + i] * B[i * n + col];
        }
    }
}

int main(void)
{
    srand(time(0));

    int m = 0, n = 0, k = 0;

    printf("Enter: m, n and k...\n");
    scanf("%d %d %d", &m, &n, &k);

    float *A = NULL;
    float *B = NULL;
    float *C = NULL;

    float *cpuA = (float *) malloc(sizeof(float) * m * k);
    float *cpuB = (float *) malloc(sizeof(float) * k * n);
    float *cpuC = (float *) malloc(sizeof(float) * m * n);

    cudaMalloc((void **) &A, sizeof(float) * m * k);
    cudaMalloc((void **) &B, sizeof(float) * k * n);
    cudaMalloc((void **) &C, sizeof(float) * m * n);

    cpuA = GPU_fill(cpuA, m, k);
    cpuB = GPU_fill(cpuB, k, n);
    cudaMemcpy(A, cpuA, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, cpuB, k * n * sizeof(float), cudaMemcpyHostToDevice);

    if (n < 5)
    {
        printf("\nFirst matrix:\n");
        print(cpuA, m, k);
        printf("Second matrix:\n");
        print(cpuB, k, n);
    }

    printf("\nMatrices have been initialized.\n");

    const int block_size = 16;
    unsigned int grid_rows = (m + block_size - 1) / block_size;
    unsigned int grid_cols = (n + block_size - 1) / block_size;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(block_size, block_size);

    float gpu_elapsed_time_ms = 0.0;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cuda_multiply<<<dimGrid, dimBlock>>>(A, B, C, m, n, k);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);

    printf("Cuda Native execution time: %f ms.\n\n", gpu_elapsed_time_ms);

    cudaMemcpy(cpuC, C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    if (n < 5)
    {
        printf("Result:\n");
        print(cpuC, m, n);
    }

    free(cpuC);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
