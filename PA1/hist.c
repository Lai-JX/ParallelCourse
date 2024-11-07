#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

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

void serial_histogram(float *array, int n, int *bins, int num_bins)
{
    int i;
    /* Initialize the bins as zero */
    for (i = 0; i < num_bins; i++) {
        bins[i] = 0; 
    }
    /* Counting */
    int idx;
    for (i = 0; i < n; i++) {
        int val = (int)array[i];
        if (val == num_bins) { /* Ensure 10 numbers go to the last bin */
            idx = num_bins - 1;
        } else {
            idx = val % num_bins;
        }
        bins[idx]++;
    }
}


void parallel_histogram(float *array, int n, int *bins, int num_bins)
{
    int i;
    /* Initialize the bins as zero */
    for (i = 0; i < num_bins; i++) {
        bins[i] = 0; 
    }
    /* Counting */
    int idx;
#   pragma omp parallel for num_threads(4)\
        private(idx) reduction(+:bins[:num_bins])
    for (i = 0; i < n; i++) {
        int val = (int)array[i];
        if (val == num_bins) { /* Ensure 10 numbers go to the last bin */
            idx = num_bins - 1;
        } else {
            idx = val % num_bins;
        }
        bins[idx]++;
    }
}

void generate_random_numbers(float *array, int n) 
{
    int i;
    float a = 10.0;
    for(i=0; i<n; ++i)
        array[i] = ((float)rand()/(float)(RAND_MAX)) * a;
}

int main(int argc, char* argv[])
{    
    int n;
    // n = strtol(argv[1], NULL, 10);
    // read number from input2.txt
    n = readNumberFromFile("input2.txt");
    int num_bins = 10;
    float *array;
    int *bins;
    int i;
    double start, end, T1, T2;
    array = (float *)malloc(sizeof(float) * n);
    bins = (int*)malloc(sizeof(int) * num_bins);
    generate_random_numbers(array, n);
    
    // personal solution
    start = omp_get_wtime();
    parallel_histogram(array, n, bins, num_bins);
    end = omp_get_wtime();
    printf("Results\n");
    for (i = 0; i < num_bins; i++) {
        printf("bins[%d]: %d\n", i, bins[i]);
    }
    T1 = (end - start) * 1000;
    printf("Running time(personal solution): %f miliseconds\n", T1);

    // serial solution
    start = omp_get_wtime();
    serial_histogram(array, n, bins, num_bins);
    end = omp_get_wtime();
    printf("Results\n");
    for (i = 0; i < num_bins; i++) {
        printf("bins[%d]: %d\n", i, bins[i]);
    }
    T2 = (end - start) * 1000;
    printf("Running time(serial solution): %f miliseconds\n", T2);

    // output
    char result[200];
    char temp[20];
    result[0] = '\0';
    for (i = 0; i < num_bins; i++) {
        sprintf(temp, "%d,", bins[i]);
        strcat(result, temp);
    }
    sprintf(temp, "%f,%f", T1, T2);
    strcat(result, temp);
    writeToFile("output2.txt", result);

    free(array);
    free(bins);
    return 0;
}

