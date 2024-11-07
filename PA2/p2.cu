#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define TILE_WIDTH 32  // 定义 CUDA 线程块的大小

// GPU 不使用共享内存的矩阵转置kernel
__global__ void matrixTranGPU(float *d_matrix, float *d_result, int Row, int Col) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int idx_x = bx * TILE_WIDTH + tx;
    int idx_y = by * TILE_WIDTH + ty;

    if (idx_x < Col && idx_y < Row) {
        d_result[idx_x * Row + idx_y] = d_matrix[idx_y * Col + idx_x];
    }
}

// GPU 使用共享内存的矩阵转置kernel
__global__ void matrixTranGPUShareMem(float *d_matrix, float *d_result, int Row, int Col) {
    __shared__ float S[TILE_WIDTH][TILE_WIDTH + 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int idx_x1 = bx * TILE_WIDTH + tx;
    int idx_y1 = by * TILE_WIDTH + ty;

    if (idx_x1 < Col && idx_y1 < Row) {
        S[ty][tx] = d_matrix[idx_y1 * Col + idx_x1];
    }
    __syncthreads();

    int idx_x2 = by * TILE_WIDTH + tx;
    int idx_y2 = bx * TILE_WIDTH + ty;
    if (idx_x2 < Row && idx_y2 < Col) {
        d_result[idx_y2 * Row + idx_x2] = S[tx][ty];
    }
}

// CPU 的矩阵转置函数
void matrixTranCPU(float *matrix, float *result, int Row, int Col) {
    for (int i = 0; i < Row; ++i) {
        for (int j = 0; j < Col; ++j) {
            result[j * Row + i] = matrix[i * Col + j];
        }
    }
}

// 计时函数
float getElapsedTime(clock_t start, clock_t end) {
    return (float)(end - start) / CLOCKS_PER_SEC * 1000.0;  // 转换为毫秒
}

// 执行矩阵转置并计时
float timeMatrixTranspose(void (*transposeFunc)(float *, float *, int, int), float *matrix, float *result, int Row, int Col, bool isCPU) {
    clock_t start = clock();
    transposeFunc(matrix, result, Row, Col);
    clock_t end = clock();
    return getElapsedTime(start, end);
}

// 执行 GPU 的矩阵转置并计时
float timeGPUTranspose(void (*kernel)(float *, float *, int, int), float *d_matrix, float *d_result, int Row, int Col) {
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((int)ceil(Col * 1.0 / TILE_WIDTH), (int)ceil(Row * 1.0 / TILE_WIDTH));
    
    cudaEvent_t startGPU, endGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&endGPU);
    
    cudaEventRecord(startGPU);
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_result, Row, Col);
    cudaEventRecord(endGPU);
    cudaEventSynchronize(endGPU);
    
    float gpuTime;
    cudaEventElapsedTime(&gpuTime, startGPU, endGPU);
    
    cudaEventDestroy(startGPU);
    cudaEventDestroy(endGPU);
    
    return gpuTime;
}

// 打印矩阵
void printMetrix(float *array, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f\t", array[i * col + j]);
        }
        printf("\n");
    }
    printf("\n\n");
}

// 验证矩阵是否一致
int checkMetrix(float *array1, float *array2, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (abs(array1[i * col + j] - array2[i * col + j]) > 1e-5) {
                return 0;
            }
        }
    }
    return 1;
}

int main() {
    // (1) 初始化
    // 读取矩阵大小
    FILE *inputFile = fopen("input2.txt", "r");
    if (!inputFile) {
        printf("无法打开 input2.txt 文件\n");
        return 1;
    }

    int Row, Col;
    fscanf(inputFile, "%d,%d", &Row, &Col);
    fclose(inputFile);

    // 分配 CPU 上的内存
    float *h_matrix = (float *)malloc(Row * Col * sizeof(float));
    float *h_resultCPU = (float *)malloc(Col * Row * sizeof(float));
    float *h_resultGPU = (float *)malloc(Col * Row * sizeof(float));

    // 初始化矩阵
    srand(time(NULL));
    for (int i = 0; i < Row * Col; ++i) {
        h_matrix[i] = (float)rand() / RAND_MAX;
    }
    // printMetrix(h_matrix,  Row,Col);

    // (2) 在 CPU 上执行矩阵转置并计时
    float cpuTime = timeMatrixTranspose(matrixTranCPU, h_matrix, h_resultCPU, Row, Col, true);
    // printMetrix(h_resultCPU, Col, Row);
    printf("Elapsed time of CPU Matrix-Transpose: %.4f ms\n\n", cpuTime);

    // (3) 在 GPU 上执行矩阵转置并计时
    float *d_matrix, *d_result;
    cudaMalloc((void **)&d_matrix, Row * Col * sizeof(float));
    cudaMalloc((void **)&d_result, Col * Row * sizeof(float));

    // 将数据从 CPU 传输到 GPU
    cudaMemcpy(d_matrix, h_matrix, Row * Col * sizeof(float), cudaMemcpyHostToDevice);

    // (4) 不使用共享内存的 GPU 矩阵转置
    float gpuTime1 = timeGPUTranspose(matrixTranGPU, d_matrix, d_result, Row, Col);
    cudaMemcpy(h_resultGPU, d_result, Row * Col * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Elapsed time of GPU Matrix-Transpose (without shared memory): %.4f ms\n", gpuTime1);
    // printMetrix(h_resultGPU, Col, Row);
    printf("Whether equal to CPU result (1:yes,0:no): %d\n\n", checkMetrix(h_resultCPU, h_resultGPU, Row, Col));

    // (5) 使用共享内存的 GPU 矩阵转置
    float gpuTime2 = timeGPUTranspose(matrixTranGPUShareMem, d_matrix, d_result, Row, Col);
    cudaMemcpy(h_resultGPU, d_result, Row * Col * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Elapsed time of GPU Matrix-Transpose (with shared memory): %.4f ms\n", gpuTime2);
    // printMetrix(h_resultGPU, Col, Row);
    printf("Whether equal to CPU result (1:yes,0:no): %d\n\n", checkMetrix(h_resultCPU, h_resultGPU, Row, Col));

    // (6) 将运行时间写入 output2.txt
    FILE *outputFile = fopen("output2.txt", "w");
    if (outputFile) {
        fprintf(outputFile, "%.2f,%.2f,%0.2f\n", cpuTime, gpuTime1, gpuTime2);
        fclose(outputFile);
    } else {
        printf("无法打开 output2.txt 文件\n");
    }

    // 释放资源
    free(h_matrix);
    free(h_resultCPU);
    free(h_resultGPU);
    cudaFree(d_matrix);
    cudaFree(d_result);

    return 0;
}
