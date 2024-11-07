#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <math.h>

#define MAX_VALUE 100.0

// mpicxx -Wall -lm -o p1 p1.c
// mpirun -n 4 --wd /home/lenovo/Lab3 --hostfile hostfile SumArrayCol

void read_sizes(int *m, int *k, int *n) {
    FILE *file = fopen("input1.txt", "r");
    fscanf(file, "%d,%d,%d", m, k, n);
    fclose(file);
}

void initialize_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = ((float)rand() / (float)(RAND_MAX)) * MAX_VALUE;
    }
}

void multiply_matrices(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i*n + j] = 0;
            for (int l = 0; l < k; l++) {
                C[i*n + j] += A[i*k + l] * B[l*n + j];
            }
        }
    }
}

void write_times(float T1, float T2) {
    FILE *file = fopen("output1.txt", "w");
    fprintf(file, "%.2f,%.2f", T1, T2);
    fclose(file);
}

int main(int argc, char *argv[]) {
    int m, k, n;
    float *A, *B, *C, *C_parallel;
    double T1, T2;
    int rows_per_process;

    // MPI variables
    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        read_sizes(&m, &k, &n);
        rows_per_process = ceil(m * 1.0 / size);
        A = (float *)malloc(size * rows_per_process * k * sizeof(float));           // size * rows_per_process: Memory alignment
        B = (float *)malloc(k * n * sizeof(float));
        C = (float *)malloc(m * n * sizeof(float));
        C_parallel = (float *)malloc(size * rows_per_process * n * sizeof(float));  // size * rows_per_process: Memory alignment

        initialize_matrix(A, m, k);
        initialize_matrix(B, k, n);

        T1 = -MPI_Wtime();
        multiply_matrices(A, B, C, m, k, n);
        T1 += MPI_Wtime();
    }

    // Broadcast m, k, n to all processes
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Distribute rows of first matrix A to all MPI processes
    rows_per_process = ceil(m * 1.0 / size);
    float *A_local = (float *)malloc(rows_per_process * k * sizeof(float));
    MPI_Scatter(A, rows_per_process * k, MPI_FLOAT, A_local, rows_per_process * k, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // All MPI processes get the full second matrix B
    if (rank != 0) {
        B = (float *)malloc(k * n * sizeof(float));
    }
    MPI_Bcast(B, k * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Each MPI process performs its part of the matrix multiplication
    float *C_local = (float *)malloc(rows_per_process * n * sizeof(float));
    T2 = -MPI_Wtime();
    multiply_matrices(A_local, B, C_local, rows_per_process, k, n);
    T2 += MPI_Wtime();

    // Gather the results from all MPI processes to form the final result matrix C_parallel
    MPI_Gather(C_local, rows_per_process * n, MPI_FLOAT, C_parallel, rows_per_process * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Only the root process performs the comparison
        for (int i = 0; i < m * n; i++) {
            if (fabs(C[i] - C_parallel[i]) > 1e-6) {
            	printf("%d,%f\n",i,fabs(C[i] - C_parallel[i]));
                printf("Results are not identical!\n");
                break;
            }
        }
	    printf("Results are identical!\n");
        write_times(T1 * 1000, T2 * 1000);
        printf("Serial running time: %f ms\n", T1 * 1000);
        printf("MPI running time: %f ms\n", T2 * 1000);
        
        free(A);
        free(B);
        free(C);
        free(C_parallel);
    }

    free(A_local);
    free(C_local);
    if (rank != 0) {
        free(B);
    }
    MPI_Finalize();

    return 0;
}

