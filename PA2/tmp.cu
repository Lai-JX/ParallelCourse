#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define TILE_WIDTH 16  // 定义 CUDA 线程块的大小

// CUDA 有效卷积核函数，使用共享内存
__global__ void convolution2D_shared(float *d_image, float *d_kernel, float *d_result, int Row, int Col, int K) {
    __shared__ float shared_image[TILE_WIDTH + 2][TILE_WIDTH + 2];  // 增加边界区域以适应卷积核
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    int kernelRadius = K / 2;
    // int newRow = Row - K + 1;  // 输出矩阵的行数
    int newCol = Col - K + 1;  // 输出矩阵的列数

    // 1. 将图像块加载到共享内存中
    if (row < Row && col < Col) {
        shared_image[ty + kernelRadius][tx + kernelRadius] = d_image[row * Col + col];

        // 处理边缘部分
        if (tx < kernelRadius && col >= kernelRadius) {
            shared_image[ty + kernelRadius][tx] = d_image[row * Col + (col - kernelRadius)];
        }
        if (tx >= blockDim.x - kernelRadius && col < Col - kernelRadius) {
            shared_image[ty + kernelRadius][tx + 2 * kernelRadius] = d_image[row * Col + (col + kernelRadius)];
        }
        if (ty < kernelRadius && row >= kernelRadius) {
            shared_image[ty][tx + kernelRadius] = d_image[(row - kernelRadius) * Col + col];
        }
        if (ty >= blockDim.y - kernelRadius && row < Row - kernelRadius) {
            shared_image[ty + 2 * kernelRadius][tx + kernelRadius] = d_image[(row + kernelRadius) * Col + col];
        }
    }

    // 同步线程，确保共享内存加载完毕
    __syncthreads();

    // 2. 执行卷积操作
    if (row >= kernelRadius && row < Row - kernelRadius && col >= kernelRadius && col < Col - kernelRadius) {
        float result = 0.0f;
        for (int i = -kernelRadius; i <= kernelRadius; i++) {
            for (int j = -kernelRadius; j <= kernelRadius; j++) {
                result += shared_image[ty + kernelRadius + i][tx + kernelRadius + j] *
                          d_kernel[(i + kernelRadius) * K + (j + kernelRadius)];
            }
        }
        // 存储卷积结果到新输出矩阵
        d_result[(row - kernelRadius) * newCol + (col - kernelRadius)] = result;
    }
}

// CPU 有效卷积函数 (去除边界处理)
void convolution2DCPU_valid(float *image, float *kernel, float *result, int Row, int Col, int K) {
    int kernelRadius = K / 2;  // 卷积核的半径
    // int newRow = Row - K + 1;  // 输出矩阵的行数
    int newCol = Col - K + 1;  // 输出矩阵的列数
    
    for (int i = kernelRadius; i < Row - kernelRadius; i++) {  // 遍历图像的有效区域
        for (int j = kernelRadius; j < Col - kernelRadius; j++) {
            float sum = 0.0f;
            for (int m = -kernelRadius; m <= kernelRadius; m++) {
                for (int n = -kernelRadius; n <= kernelRadius; n++) {
                    int curRow = i + m;
                    int curCol = j + n;
                    sum += image[curRow * Col + curCol] * kernel[(m + kernelRadius) * K + (n + kernelRadius)];
                }
            }
            // 将结果存储到新的输出矩阵中
            result[(i - kernelRadius) * newCol + (j - kernelRadius)] = sum;
        }
    }
}

// 计时函数
float getElapsedTime(clock_t start, clock_t end) {
    return (float)(end - start) / CLOCKS_PER_SEC * 1000.0f;  // 转换为毫秒
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

// 主函数
int main() {
    // (1) 读取图像和卷积核的大小
    FILE *inputFile = fopen("input3.txt", "r");
    if (!inputFile) {
        printf("无法打开 input3.txt 文件\n");
        return 1;
    }

    int Row, Col, K;
    fscanf(inputFile, "%d,%d,%d", &Row, &Col, &K);
    fclose(inputFile);

    // (2) 初始化图像和卷积核
    float *h_image = (float *)malloc(Row * Col * sizeof(float));
    float *h_kernel = (float *)malloc(K * K * sizeof(float));
    int newRow = Row - K + 1;
    int newCol = Col - K + 1;
    float *h_resultCPU = (float *)malloc(newRow * newCol * sizeof(float));
    float *h_resultGPU = (float *)malloc(newRow * newCol * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < Row * Col; i++) {
        h_image[i] = (float)(rand() % 256);  // 随机初始化图像
    }
    for (int i = 0; i < K * K; i++) {
        h_kernel[i] = (float)(rand() % 9);  // 随机初始化卷积核
    }
    printMetrix(h_image, Row, Col);
    printMetrix(h_kernel, K, K);

    // (3) 在 CPU 上执行卷积并计时 (有效卷积)
    clock_t startCPU = clock();
    convolution2DCPU_valid(h_image, h_kernel, h_resultCPU, Row, Col, K);
    clock_t endCPU = clock();
    float cpuTime = getElapsedTime(startCPU, endCPU);
    printf("CPU 时间: %.2f ms\n", cpuTime);
    printMetrix(h_resultCPU, newRow, newCol);

    // (4) 在 GPU 上执行卷积并计时 (共享内存优化)
    float *d_image, *d_kernel, *d_result;
    cudaMalloc((void **)&d_image, Row * Col * sizeof(float));
    cudaMalloc((void **)&d_kernel, K * K * sizeof(float));
    cudaMalloc((void **)&d_result, newRow * newCol * sizeof(float));

    // 将数据从 CPU 传输到 GPU
    cudaMemcpy(d_image, h_image, Row * Col * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, K * K * sizeof(float), cudaMemcpyHostToDevice);

    // 设置 CUDA 线程块和网格大小
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((int)ceil(Col * 1.0 / TILE_WIDTH), (int)ceil(Row * 1.0 / TILE_WIDTH));

    cudaEvent_t startGPU, endGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&endGPU);

    cudaEventRecord(startGPU);
    convolution2D_shared<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_kernel, d_result, Row, Col, K);
    cudaEventRecord(endGPU);
    cudaEventSynchronize(endGPU);

    // 将结果从 GPU 传输回 CPU
    cudaMemcpy(h_resultGPU, d_result, newRow * newCol * sizeof(float), cudaMemcpyDeviceToHost);

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, startGPU, endGPU);
    printf("GPU 时间: %.2f ms\n", gpuTime);
    printMetrix(h_resultGPU, newRow, newCol);

    cudaEventDestroy(startGPU);
    cudaEventDestroy(endGPU);

    // (5) 验证 GPU 结果是否与 CPU 一致
    int correct = 1;
    for (int i = 0; i < newRow * newCol; i++) {
        if (abs(h_resultCPU[i] - h_resultGPU[i]) > 1e-5) {
            correct = 0;
            break;
        }
    }
    if (correct) {
        printf("GPU 结果与 CPU 结果一致。\n");
    } else {
        printf("GPU 结果与 CPU 结果不一致！\n");
    }

    // (6) 将 CPU 和 GPU 的运行时间写入 output3.txt
    FILE *outputFile = fopen("output3.txt", "w");
    if (outputFile) {
        fprintf(outputFile, "%.2f,%.2f\n", cpuTime, gpuTime);
        fclose(outputFile);
    } else {
        printf("无法打开 output3.txt 文件\n");
    }

    // 释放内存
    free(h_image);
    free(h_kernel);
    free(h_resultCPU);
    free(h_resultGPU);
    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_result);

    return 0;
}
