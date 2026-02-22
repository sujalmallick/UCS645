#include <cmath>
#include <omp.h>
#include <vector>
using namespace std;

/*
Sequential baseline
*/
void correlate_seq(int ny, int nx, const float* data, float* result) {

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {

            double sum_i = 0.0, sum_j = 0.0;
            double sum_i2 = 0.0, sum_j2 = 0.0;
            double sum_ij = 0.0;

            for (int x = 0; x < nx; x++) {
                double a = data[x + i*nx];
                double b = data[x + j*nx];

                sum_i += a;
                sum_j += b;
                sum_i2 += a*a;
                sum_j2 += b*b;
                sum_ij += a*b;
            }

            double numerator = nx * sum_ij - sum_i * sum_j;
            double denom_i = nx * sum_i2 - sum_i * sum_i;
            double denom_j = nx * sum_j2 - sum_j * sum_j;

            double denominator = sqrt(denom_i * denom_j);

            result[i + j*ny] =
                (denominator == 0.0) ? 0.0 : numerator / denominator;
        }
    }
}

/*
Parallel OpenMP version
*/
void correlate(int ny, int nx, const float* data, float* result) {

#pragma omp parallel for schedule(static)
    for (int i = 0; i < ny; i++) {

        for (int j = 0; j <= i; j++) {

            double sum_i = 0.0, sum_j = 0.0;
            double sum_i2 = 0.0, sum_j2 = 0.0;
            double sum_ij = 0.0;

            for (int x = 0; x < nx; x++) {
                double a = data[x + i*nx];
                double b = data[x + j*nx];

                sum_i += a;
                sum_j += b;
                sum_i2 += a*a;
                sum_j2 += b*b;
                sum_ij += a*b;
            }

            double numerator = nx * sum_ij - sum_i * sum_j;
            double denom_i = nx * sum_i2 - sum_i * sum_i;
            double denom_j = nx * sum_j2 - sum_j * sum_j;

            double denominator = sqrt(denom_i * denom_j);

            result[i + j*ny] =
                (denominator == 0.0) ? 0.0 : numerator / denominator;
        }
    }
}
