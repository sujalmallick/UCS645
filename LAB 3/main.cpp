#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace std;

void correlate(int ny, int nx, const float* data, float* result);
void correlate_seq(int ny, int nx, const float* data, float* result);

int main(int argc, char* argv[]) {

    if (argc < 3) {
        cout << "Usage: ./correlate ny nx\n";
        return 0;
    }

    int ny = atoi(argv[1]);
    int nx = atoi(argv[2]);

    vector<float> data(ny * nx);
    vector<float> result(ny * ny);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0, 1.0);

    for (int i = 0; i < ny * nx; i++)
        data[i] = dis(gen);

    cout << "Matrix size: " << ny << " x " << nx << endl;

    // Sequential
    auto start = chrono::high_resolution_clock::now();
    correlate_seq(ny, nx, data.data(), result.data());
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> seq_time = end - start;

    cout << "Sequential Time: " << seq_time.count() << " s\n";

    // Parallel
    start = chrono::high_resolution_clock::now();
    correlate(ny, nx, data.data(), result.data());
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> par_time = end - start;

    cout << "Parallel Time: " << par_time.count() << " s\n";

    cout << "Speedup: " << seq_time.count() / par_time.count() << endl;

    return 0;
}
