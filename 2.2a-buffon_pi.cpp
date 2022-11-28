#include <iomanip>
#include <iostream>
#include <cmath>
#include <cstddef>
#include <random>

#include <omp.h>

#include "vectors_and_matrices/array_types.hpp"

using std::cin;
using std::cout;
using ptrdiff_t = std::ptrdiff_t;
using size_t = std::size_t;

const double PI = 3.141592653589793;

double mc_pi(ptrdiff_t niter, size_t seed)
{
    double num_crosses = 0;
    std::mt19937_64 rng(seed);
    std::normal_distribution<> rand_nrm(0.0, 1.0);
    std::uniform_real_distribution<double> rand_un(-1.0, 1.0);
#pragma omp parallel
    {
        #pragma omp for reduction(+: num_crosses)
        for (ptrdiff_t i = 0; i < niter; ++i)
        {
            // generate a unit vector with a uniform rotation
            double x = rand_nrm(rng), y = rand_nrm(rng);
            double l = std::hypot(x, y);
            
            y *= 0.5 / l;

            // check if a horizontal line crosses a needle
            double y_line = rand_un(rng);
            num_crosses += std::abs(y_line) < std::abs(y);
        }
    }
    // p = 2L / (r * pi) = 1 / pi if 2L = r
    // r is width of uniform distribution (2 if it is from -1 to 1)
    // L is length of the needle (1 in our case)
    double pi_est = niter / num_crosses;
    return pi_est;
}

int main(int argc, char** argv)
{
    ptrdiff_t niter;

    cin >> niter;

    double t1 = omp_get_wtime();

    double pi_est = mc_pi(niter, 1234);

    double t2 = omp_get_wtime();
    
    cout << "Computed pi: " << std::setprecision(16) << pi_est << std::endl;
    cout << "Exact pi: " << std::setprecision(16) << PI << std::endl;
    cout << "Time: " << t2 - t1 << std::endl;

    return 0;
}