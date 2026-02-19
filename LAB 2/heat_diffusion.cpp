#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

int main() {

    omp_set_dynamic(0);

    int N = 2000;
    int T = 100;

    vector<vector<double>> A(N, vector<double>(N, 25.0));
    vector<vector<double>> B(N, vector<double>(N, 25.0));

    vector<int> threads = {1,2,4,8};
    double serial_time = 0;

    cout << "\nHeat Diffusion Simulation\n";
    cout << "Grid Size: " << N << " x " << N << "\n\n";

    for(int t : threads) {

        omp_set_num_threads(t);

        double start = omp_get_wtime();

        for(int step = 0; step < T; step++) {

            #pragma omp parallel for collapse(2) schedule(static)
            for(int i = 1; i < N-1; i++) {
                for(int j = 1; j < N-1; j++) {
                    B[i][j] = 0.25 * (
                        A[i+1][j] +
                        A[i-1][j] +
                        A[i][j+1] +
                        A[i][j-1]
                    );
                }
            }

            swap(A, B);
        }

        double end = omp_get_wtime();
        double time = end - start;

        if(t==1) serial_time = time;

        double speedup = serial_time / time;
        double efficiency = speedup / t;

        cout << "Threads: " << t
             << "  Time: " << time
             << "  Speedup: " << speedup
             << "  Efficiency: " << efficiency << endl;
    }

    return 0;
}
