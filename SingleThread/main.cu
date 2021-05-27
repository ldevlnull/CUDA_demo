#include <stdio.h>
#include <time.h>
#include <stdlib.h>

void init(float *matrix, const int cols, const int rows)
{
    int i;
    for (i = 0; i < rows * cols; ++i)
    {
        matrix[i] = rand() % 10;
    }
}


void multiply(float *A, float *B, float *C, const int m, const int n, const int k)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            C[i * n + j] = 0;
            for (int h = 0; h < k; ++h)
            {
                C[i * n + j] += A[i * k + h] * B[h * n + j];
            }
        }
    }
}

void print(float *matrix, const int cols, const int rows)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            printf("%f\t", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}


int main()
{
    srand(time(0));

    int m = 0;
    int n = 0;
    int k = 0;

    printf("Enter: m, n and k...\n");
    scanf("%d %d %d", &m, &n, &k);

    float *matrix_A = (float *) malloc(sizeof(float) * m * k);
    float *matrix_B = (float *) malloc(sizeof(float) * k * n);
    float *result = (float *) malloc(sizeof(float) * m * n);

    init(matrix_A, m, k);
    init(matrix_B, k, n);

    if (m < 5)
    {
        printf("\nMatrix A\n");
        print(matrix_A, m, k);

        printf("Matrix B\n");
        print(matrix_B, k, n);
    }

    printf("\nMatrices have been initialized.\n");

    float cpu_elapsed_time_ms = 0;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    multiply(matrix_A, matrix_B, result, m, n, k);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);

    printf("CPU execution time: %f ms.\n\n", cpu_elapsed_time_ms);

    if (m < 5)
    {
        printf("Result matrix\n");
        print(result, m, n);
    }
    free(matrix_A);
    free(matrix_B);
    free(result);

    return 0;
}
