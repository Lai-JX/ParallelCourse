#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int _print_matrix(int N, double **C) {
    int i,j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%f\t", C[i][j]);
        }
        printf("\n");
    }
}

// 封装读取文件数字的函数
int readNumberFromFile(const char *filePath) {
    FILE *file;
    int number;

    // 打开文件
    file = fopen(filePath, "r");
    if (file == NULL) {
        // 如果文件打开失败，返回错误码 -1
        printf("无法打开文件: %s\n", filePath);
        return -1;
    }

    // 从文件中读取数字
    if (fscanf(file, "%d", &number) != 1) {
        // 如果读取失败，返回错误码 -1
        printf("读取数字失败\n");
        fclose(file);  // 关闭文件
        return -1;
    }

    // 关闭文件
    fclose(file);
    return number;
}

// 动态分配二维数组的函数
double **allocateDoubleArray(int rows, int cols) {
    // 为行指针数组分配空间
    double **array = (double **)malloc(rows * sizeof(double *));
    if (array == NULL) {
        return NULL;
    }

    // 为每一行的列分配空间
    for (int i = 0; i < rows; i++) {
        array[i] = (double *)malloc(cols * sizeof(double));
        if (array[i] == NULL) {
            // 如果分配失败，记得释放已经分配的内存
            for (int j = 0; j < i; j++) {
                free(array[j]);
            }
            free(array);
            return NULL;
        }
    }
    return array;
}

// 封装函数，将字符串写入指定文件
void writeToFile(const char *path, const char *string) {
    // 打开文件以写模式 ("w")
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        printf("无法打开文件: %s\n", path);
        return;
    }

    // 将字符串写入文件
    fprintf(file, "%s", string);

    // 关闭文件
    fclose(file);

}

int main(int argc, char *argv) {
    int i, j, k, N;
    double sum;
    double start, end, T1, T2;
    double **A, **B, **C;
    // read N from input1.txt
    N = readNumberFromFile("input1.txt");

    // allocate memory for matrix A, B, C
    A = allocateDoubleArray(N, N);
    B = allocateDoubleArray(N, N);
    C = allocateDoubleArray(N, N);
    if (A == NULL || B == NULL || C == NULL) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // init metrix A, B, C
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = j*1;
            B[i][j] = i*j+2;
            C[i][j] = j-i*2;
        }
    }

    // personal solution
    start = omp_get_wtime(); //start time measurement
#   pragma omp parallel for num_threads(4)\
        reduction(+:sum) private(i,j,k) 
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum = 0;
            for (k=0; k < N; k++) {
                sum += A[i][k]*B[k][j];
            }
            C[i][j] = sum;
        }
    }
    end = omp_get_wtime(); //end time measurement
    T1 = (end - start) * 1000;
    printf("Time of computation(personal solution): %f miliseconds\n", T1);

    // init metrix A, B, C
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = j*1;
            B[i][j] = i*j+2;
            C[i][j] = j-i*2;
        }
    }

    // serial solution
    start = omp_get_wtime(); //start time measurement
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum = 0;
            for (k=0; k < N; k++) {
                sum += A[i][k]*B[k][j];
            }
            C[i][j] = sum;
        }
    }
    end = omp_get_wtime(); //end time measurement
    T2 = (end - start) * 1000;
    printf("Time of computation(serial solution): %f miliseconds\n", T2);

    // output
    char result[100];
    sprintf(result, "%f,%f", T1, T2);
    writeToFile("output1.txt", result);
    return(0);
}