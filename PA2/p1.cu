#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// #define BLOCK_SIZE 16  // 定义 CUDA 线程块的大小
#define TILE_WIDTH 32       // 定义 CUDA 线程块的大小

// CUDA 核函数用于矩阵-向量乘法
__global__ void matVecMulKernel(float *d_matrix, float *d_vector, float *d_result, int Row, int Col) {
    /*
        例：输入（4，5）
        Row=4, Col=5
        4行5列
        行的长度：5
        列的长度：4


     */
    __shared__ float ds_v[TILE_WIDTH];

    int bx = blockIdx.x;

    int tx = threadIdx.x;

    int row = bx * TILE_WIDTH + tx;

    float pValue = 0.0;

    for(int t = 0; t < (int)ceil(Col*1.0 / TILE_WIDTH); ++t)
    {
        if(t * TILE_WIDTH + tx < Col)
            ds_v[tx] = d_vector[t * TILE_WIDTH + tx];
        else
            ds_v[tx] = 0;
        __syncthreads();

        for(int i = 0; i < TILE_WIDTH; ++i) {
            if (row < Row && t * TILE_WIDTH + i < Col)
                pValue += d_matrix[row * Col + t * TILE_WIDTH + i] * ds_v[i];
        }
        __syncthreads();
    }
    if(row < Row)
        d_result[row] = pValue;
}

// CPU 上的矩阵-向量乘法
void matVecMulCPU(float *matrix, float *vector, float *result, int Row, int Col) {
    for (int i = 0; i < Row; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < Col; ++j) {
            result[i] += matrix[i * Col + j] * vector[j];
        }
    }
}

// 计时函数
float getElapsedTime(clock_t start, clock_t end) {
    return (float)(end - start) / CLOCKS_PER_SEC * 1000.0;  // 转换为毫秒
}

void printArray(float *array, int n) {
    for(int i=0; i<n; i++) {
        printf("%f\t", array[i]);
    }
    printf("\n\n");
}

int main() {
    // (1) 从 input1.txt 读取矩阵大小
    FILE *inputFile = fopen("input1.txt", "r");
    if (!inputFile) {
        printf("无法打开 input1.txt 文件\n");
        return 1;
    }

    int Row, Col;
    fscanf(inputFile, "%d,%d", &Row, &Col);
    fclose(inputFile);

    // 分配 CPU 上的内存
    float *h_matrix = (float *)malloc(Row * Col * sizeof(float));
    float *h_vector = (float *)malloc(Col * sizeof(float));
    float *h_resultCPU = (float *)malloc(Row * sizeof(float));
    float *h_resultGPU = (float *)malloc(Row * sizeof(float));

    // (2) 初始化矩阵和向量
    srand(time(NULL));
    for (int i = 0; i < Row * Col; ++i) {
        h_matrix[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < Col; ++i) {
        h_vector[i] = (float)rand() / RAND_MAX;
    }

    // (3) 在 CPU 上执行矩阵-向量乘法并测量时间
    clock_t startCPU = clock();
    matVecMulCPU(h_matrix, h_vector, h_resultCPU, Row, Col);
    clock_t endCPU = clock();
    float cpuTime = getElapsedTime(startCPU, endCPU);
    // printArray(h_resultCPU, Row);
    printf("elapsed time of cpu Matrix-Vector Multiplication : %.4f ms\n", cpuTime);
    

    // (4) 在 GPU 上执行矩阵-向量乘法并测量时间
    float *d_matrix, *d_vector, *d_result;
    cudaMalloc((void **)&d_matrix, Row * Col * sizeof(float));
    cudaMalloc((void **)&d_vector, Col * sizeof(float));
    cudaMalloc((void **)&d_result, Row * sizeof(float));

    // 将数据从 CPU 传输到 GPU
    cudaMemcpy(d_matrix, h_matrix, Row * Col * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, Col * sizeof(float), cudaMemcpyHostToDevice);

    // 设置 CUDA 线程块和网格大小
    dim3 threadsPerBlock(TILE_WIDTH);
    dim3 blocksPerGrid((int)ceil(Row*1.0 / TILE_WIDTH));

    cudaEvent_t startGPU, endGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&endGPU);

    cudaEventRecord(startGPU);
    matVecMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_vector, d_result, Row, Col);
    cudaEventRecord(endGPU);
    cudaEventSynchronize(endGPU);

    // 将结果从 GPU 传输回 CPU
    cudaMemcpy(h_resultGPU, d_result, Row * sizeof(float), cudaMemcpyDeviceToHost);
    // printArray(h_resultGPU, Row);
    

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, startGPU, endGPU);
    printf("elapsed time of gpu Matrix-Vector Multiplication: %.4f ms\n", gpuTime);


    // (5) 比较 CPU 和 GPU 的结果
    int match = 1;
    for (int i = 0; i < Row; ++i) {
        // printf("%f\n", abs(h_resultCPU[i] - h_resultGPU[i]));
        if (abs(h_resultCPU[i] - h_resultGPU[i]) > 1e-2) {
            match = 0;
            break;
        }
    }

    if (match) {
        printf("CPU 和 GPU 结果一致\n");
    } else {
        printf("CPU 和 GPU 结果不一致\n");
    }

    // (6) 将运行时间写入 output1.txt
    FILE *outputFile = fopen("output1.txt", "w");
    if (outputFile) {
        fprintf(outputFile, "%.2f,%.2f\n", cpuTime, gpuTime);
        fclose(outputFile);
    } else {
        printf("无法打开 output1.txt 文件\n");
    }

    // 释放资源
    free(h_matrix);
    free(h_vector);
    free(h_resultCPU);
    free(h_resultGPU);
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);
    cudaEventDestroy(startGPU);
    cudaEventDestroy(endGPU);

    return 0;
}
