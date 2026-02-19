#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>
#include <random>

using namespace std;

const double EPSILON = 1.0;
const double SIGMA = 1.0;

struct Vector3 {
    double x, y, z;
};

void reset_forces(vector<Vector3>& f) {
    for(int i = 0; i < f.size(); i++) {
        f[i].x = 0.0;
        f[i].y = 0.0;
        f[i].z = 0.0;
    }
}

void init_particles(int n, vector<Vector3>& pos) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 10.0);

    for(int i = 0; i < n; i++) {
        pos[i].x = dis(gen);
        pos[i].y = dis(gen);
        pos[i].z = dis(gen);
    }
}

int main() {

    omp_set_dynamic(0);   // Disable dynamic adjustment

    int n = 2000;

    vector<Vector3> pos(n);
    vector<Vector3> f(n);

    init_particles(n, pos);

    vector<int> threads = {1, 2, 4, 8};
    double serial_time = 0.0;

    cout << "Molecular Dynamics Force Calculation\n";
    cout << "Particles: " << n << "\n\n";

    cout << left << setw(10) << "Threads"
         << setw(15) << "Energy"
         << setw(15) << "Time (s)"
         << setw(12) << "Speedup"
         << setw(12) << "Efficiency" << endl;

    for(int t : threads) {

        omp_set_num_threads(t);

        reset_forces(f);

        double energy = 0.0;
        double start = omp_get_wtime();

        #pragma omp parallel
        {
            #pragma omp single
            cout << "\nRunning with " << omp_get_num_threads() << " threads\n";
        }

        #pragma omp parallel for schedule(dynamic,10) reduction(+:energy)
        for(int i = 0; i < n; i++) {

            double fx = 0.0, fy = 0.0, fz = 0.0;

            for(int j = i + 1; j < n; j++) {

                double dx = pos[i].x - pos[j].x;
                double dy = pos[i].y - pos[j].y;
                double dz = pos[i].z - pos[j].z;

                double r2 = dx*dx + dy*dy + dz*dz;
                if(r2 < 1e-4) continue;

                double r2inv = 1.0 / r2;
                double r6inv = r2inv * r2inv * r2inv;
                double r12inv = r6inv * r6inv;

                double fscalar = (24.0 * EPSILON / r2) * (2.0 * r12inv - r6inv);
                energy += 4.0 * EPSILON * (r12inv - r6inv);

                double f_x = fscalar * dx;
                double f_y = fscalar * dy;
                double f_z = fscalar * dz;

                fx += f_x;
                fy += f_y;
                fz += f_z;

                #pragma omp atomic
                f[j].x -= f_x;
                #pragma omp atomic
                f[j].y -= f_y;
                #pragma omp atomic
                f[j].z -= f_z;
            }

            #pragma omp atomic
            f[i].x += fx;
            #pragma omp atomic
            f[i].y += fy;
            #pragma omp atomic
            f[i].z += fz;
        }

        double end = omp_get_wtime();
        double time = end - start;

        if(t == 1) serial_time = time;

        double speedup = serial_time / time;
        double eff = speedup / t;

        cout << left << setw(10) << t
             << setw(15) << scientific << setprecision(5) << energy
             << setw(15) << fixed << setprecision(5) << time
             << setw(12) << setprecision(2) << speedup
             << setw(12) << setprecision(2) << eff << endl;
    }

    cout << string(64, '-') << endl;

    return 0;
}
