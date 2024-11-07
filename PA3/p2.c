#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#define MAX_VALUE 100.0
#define MAX 0
#define SUM 1

int max(int a, int b) {
    return a > b ? a : b;
}

int min(int a, int b) {
    return a < b ? a : b;
}

// Ring-based Allreduce implementation
void RING_Allreduce(float *sendbuf, float *recvbuf, int count, int op, MPI_Comm comm) {
    int size, rank;
    int send_to, send_offset, send_count, recv_from, recv_offset, recv_count;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    int n_per_process = ceil(count * 1.0 / size);
    // Initialize recvbuf with sendbuf values
    for (int i = 0; i < count; i++) {
        recvbuf[i] = sendbuf[i];
    }

    float *tempbuf = (float*) malloc(n_per_process * sizeof(float));

    for (int step = 0; step < size-1; step++) {
        send_to = (rank + 1) % size;
        send_offset = (rank - step + size) % size * n_per_process;
        send_count = min(max(count - send_offset, 0), n_per_process);
        recv_from = (rank - 1 + size) % size;
        recv_offset = (rank - step -1 + size) % size * n_per_process;
        recv_count = min(max(count - recv_offset, 0), n_per_process);

        // Send recvbuf and receive tempbuf
        MPI_Sendrecv(recvbuf+send_offset, send_count, MPI_FLOAT, send_to, 0,
                     tempbuf, recv_count, MPI_FLOAT, recv_from, 0,
                     comm, MPI_STATUS_IGNORE);

        // Perform the specified operation (SUM or MAX)
        for (int i = 0; i < recv_count; i++) {
            if (op == SUM) {
                recvbuf[i+recv_offset] += tempbuf[i];
            } else if (op == MAX) {
                recvbuf[i+recv_offset] = (recvbuf[i+recv_offset] > tempbuf[i]) ? recvbuf[i+recv_offset] : tempbuf[i];
            }
        }
    }
    MPI_Barrier(comm);
    for (int step = 0; step < size-1; step++) {
        send_to = (rank + 1) % size;
        send_offset = (rank - step + size + 1) % size * n_per_process;
        send_count = min(max(count - send_offset, 0), n_per_process);
        recv_from = (rank - 1 + size) % size;
        recv_offset = (rank - step + size) % size * n_per_process;
        recv_count = min(max(count - recv_offset, 0), n_per_process);

        // Send recvbuf and receive tempbuf
        MPI_Sendrecv(recvbuf+send_offset, send_count, MPI_FLOAT, send_to, 0,
                     recvbuf+recv_offset, recv_count, MPI_FLOAT, recv_from, 0,
                     comm, MPI_STATUS_IGNORE);
    }
    free(tempbuf);
}

int main(int argc, char *argv[]) {
    int rank, size, n=0, op_type=-1;
    float *array, *mpi_result, *ring_result;
    double mpi_time, ring_time;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Input array size and operation type from user
    if (rank == 0) {

        while (n <=0) {
            printf("Enter the size of the array(larger than process num): \n");
            scanf("%d", &n);
        }

        while (op_type != 0 && op_type !=1) {
            printf("Enter the operation (0 for MAX, 1 for SUM): \n");
            scanf("%d", &op_type);
        }
        // assert(n >= size);
    }
    
    // Broadcast array size and operation type to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&op_type, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for the arrays
    array = (float *)malloc(n * sizeof(float));
    mpi_result = (float *)malloc(n * sizeof(float));
    ring_result = (float *)malloc(n * sizeof(float));

    // Initialize array with random values
    srand(time(NULL) + rank);
    for (int i = 0; i < n; i++) {
        array[i] = ((float)rand() / (float)(RAND_MAX)) * MAX_VALUE;
    }

    // Perform MPI_Allreduce
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes are synchronized
    mpi_time = -MPI_Wtime();
    MPI_Allreduce(array, mpi_result, n, MPI_FLOAT, (op_type == SUM) ? MPI_SUM : MPI_MAX, MPI_COMM_WORLD);
    mpi_time += MPI_Wtime();

    // Perform RING_Allreduce
    MPI_Barrier(MPI_COMM_WORLD);
    ring_time = -MPI_Wtime();
    RING_Allreduce(array, ring_result, n, op_type, MPI_COMM_WORLD);
    ring_time += MPI_Wtime();

    // Verify the results
    if (rank == 0) {
        int correct = 1;
        for (int i = 0; i < n; i++) {
            if (fabs(ring_result[i] - mpi_result[i]) > 1e-3) {
                printf("%d:%f,%f\n", i, ring_result[i], mpi_result[i]);
                printf("incorrect\n");
                correct = 0;
                break;
            }
        }
        printf("Array size: %d\n", n);
        printf("MPI_Allreduce time: %f ms\n", mpi_time * 1000);
        printf("RING_Allreduce time: %f ms\n", ring_time * 1000);
        if (correct) {
            printf("The results match!\n");
        } else {
            printf("The results do not match!\n");
        }
    }

    // Clean up
    free(array);
    free(mpi_result);
    free(ring_result);

    MPI_Finalize();
    return 0;
}
