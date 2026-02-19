#include <iostream>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <random>

using namespace std;

const int MATCH = 2;
const int MISMATCH = -1;
const int GAP = -1;

int main() {

    int N = 2000;   // sequence length
    int M = 2000;

    omp_set_dynamic(0);

    // Generate random DNA sequences
    string bases = "ACGT";
    string s1(N, 'A'), s2(M, 'A');

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0,3);

    for(int i=0;i<N;i++) s1[i] = bases[dis(gen)];
    for(int j=0;j<M;j++) s2[j] = bases[dis(gen)];

    vector<vector<int>> H(N+1, vector<int>(M+1, 0));

    vector<int> threads = {1,2,4,8};
    double serial_time = 0;

    cout << "\nSmith-Waterman Local Alignment\n";
    cout << "Matrix Size: " << N << " x " << M << "\n\n";

    for(int t : threads) {

        omp_set_num_threads(t);

        double start = omp_get_wtime();

        // Wavefront Parallelization
        for(int k = 2; k <= N+M; k++) {

            #pragma omp parallel for schedule(static)
            for(int i = max(1, k-M); i <= min(N, k-1); i++) {

                int j = k - i;

                int score_diag = H[i-1][j-1] + 
                    (s1[i-1] == s2[j-1] ? MATCH : MISMATCH);

                int score_up = H[i-1][j] + GAP;
                int score_left = H[i][j-1] + GAP;

                H[i][j] = max({0, score_diag, score_up, score_left});
            }
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
