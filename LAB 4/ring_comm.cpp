#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int rank, size, value;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    if (rank == 0) {
        value = 100;
        printf("Process 0 sending value: %d\n", value);
        MPI_Send(&value, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        MPI_Recv(&value, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 0 final received value: %d\n", value);
    } else {
        MPI_Recv(&value, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received: %d", rank, value);
        value += rank;
        printf(", adding rank %d, sending: %d\n", rank, value);
        MPI_Send(&value, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
