#include <stdlib.h>
#include <curand.h>
#include <cublas_v2.h>
#include <stdio.h>

void print(const float *A, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            printf("%f\t", A[j * rows + i]);
        }
        printf("\n");
    }
    printf("\n");
}

float *GPU_fill(float *matrix, int rows, int cols)
{
    int i;
    for (i = 0; i < rows * cols; ++i)
    {
        matrix[i] = rand() % 10;
    }
    return matrix;
}


void cublas_multiply(const float *A, const float *B, float *C, const int m, const int k, const int n)
{
    int lda = m, ldb = k, ldc = m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    cublasHandle_t handle;
    cublasCreate(&handle);

    float gpu_elapsed_time_ms = 0.0;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);

    printf("cuBlas execution time: %f ms.\n\n", gpu_elapsed_time_ms);

    cublasDestroy(handle);
}

int main()
{
    int m = 0, n = 0, k = 0;

    printf("Enter: m, n, k\n");
    scanf("%d %d %d", &m, &n, &k);

    int rows_A, cols_A, rows_B, cols_B, rows_C, cols_C;

    rows_A = m;
    cols_A = n;
    rows_B = n;
    cols_B = k;
    rows_C = m;
    cols_C = k;

    float *h_A = (float *) malloc(rows_A * cols_A * sizeof(float));
    float *h_B = (float *) malloc(rows_B * cols_B * sizeof(float));
    float *h_C = (float *) malloc(rows_C * cols_C * sizeof(float));

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, rows_A * cols_A * sizeof(float));
    cudaMalloc(&d_B, rows_B * cols_B * sizeof(float));
    cudaMalloc(&d_C, rows_C * cols_C * sizeof(float));

    h_A = GPU_fill(h_A, rows_A, cols_A);
    h_B = GPU_fill(h_B, rows_B, cols_B);

    cudaMemcpy(d_A, h_A, rows_A * cols_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, rows_B * cols_B * sizeof(float), cudaMemcpyHostToDevice);

    if (m < 5)
    {
        printf("\nFirst matrix\n");
        print(h_A, rows_A, cols_A);
        printf("Second matrix\n");
        print(h_B, rows_B, cols_B);
    }

    printf("\nMatrices have been initialized.\n");

    cublas_multiply(d_A, d_B, d_C, rows_A, cols_A, cols_B);
    cudaMemcpy(h_C, d_C, rows_C * cols_C * sizeof(float), cudaMemcpyDeviceToHost);

    if (m < 5)
    {
        printf("Result matrix\n");
        print(h_C, rows_C, cols_C);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}