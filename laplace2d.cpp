#include <iostream>
#include <memory>
#include <cstdint>
#include <iomanip>
#include <random>
#include <cmath>
#include "vectors_and_matrices/array_types.hpp"

#include <omp.h>

using ptrdiff_t = std::ptrdiff_t;
using size_t = std::size_t;

void random_filling(matrix<double> u, double ampl, size_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist_charge(-ampl, ampl);
    for (ptrdiff_t i = 1; i < u.nrows() - 1; i++)
	{
		for (ptrdiff_t j = 1; j < u.ncols() - 1; j++)
			u(i, j) = dist_charge(rng);
	}
}
void single_filling(matrix<double> u) {
    for (ptrdiff_t i = 0; i < u.nrows(); i++)
	{
		for (ptrdiff_t j = 0; j < u.ncols(); j++)
			u(i, j) = 1.0;
	}
}
void random_bounds(matrix<double> u, ptrdiff_t ncenters, double ampl, size_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist_charge(-ampl, ampl);
    std::uniform_real_distribution<double> dist_coord(-1, 1);
    ptrdiff_t generated=0, nx = u.nrows(), ny = u.ncols();
    double hx = 1.0 / (nx-1), hy = 1.0 / (ny -1);
    while (generated < ncenters)
    {
        double x = dist_coord(rng), y = dist_coord(rng);
        if ((std::abs(x) > 0.5) && (std::abs(y) > 0.5))
        {
            double charge = dist_charge(rng);
            for (ptrdiff_t i=0; i < nx; i++)
            {
                u(i, 0) = 0;
                u(i, ny-1) = 0;
            }
            for (ptrdiff_t j=1; j < ny-1; j++)
            {
                u(0, j) = 0;
                u(nx-1, j) = 0;
            }
            for (ptrdiff_t i=0; i < nx; i++)
            {
                double r = hypot(x - (-0.5 + i * hx), y + 0.5);
                u(i, 0) += charge / r;
                r = hypot(x - (-0.5 + i * hx), y - 0.5);
                u(i, ny-1) += charge / r;
            }
            for (ptrdiff_t j=1; j < ny-1; j++)
            {
                double r = hypot(x + 0.5, y - (-0.5 + j * hy));
                u(0, j) += charge / r;
                r = hypot(x - 0.5, y - (-0.5 + j * hy));
                u(nx-1, j) += charge / r;
            }
            generated += 1;
        }
    }
}
int test_laplace(matrix<double> u, double atol)
{
    ptrdiff_t nx = u.nrows(), ny = u.ncols();
    for (ptrdiff_t i = 1; i < nx-1; i++)
    {
        for (ptrdiff_t j = 1; j < ny-1; j++)
        {
            if (std::fabs(u(i-1, j) + u(i+1, j) + u(i, j-1) + u(i, j+1) - 4 * u(i, j)) > atol)
            {
                return 0;
            }
        }
    }
    return 1;
}
void laplace2d(matrix<double> u, double hx, double hy)
{
    ptrdiff_t j;
    matrix<double> u_next = u;

    while(!test_laplace(u, 1e-6)) {
        #pragma omp parallel for private(j)
        for (ptrdiff_t i = 0; i < u.nrows() - 1; i++) {
            for (j = 0; j < u.ncols() - 1; j++) {
                u_next(i, j) = 0.25 * (u(i-1, j) + u(i+1,j) + u(i,j-1) + u(i, j+1));
            }
        }
        #pragma omp critical
        {
            u = u_next;
        }
    }
}

int main(int argc, char* argv[])
{
    ptrdiff_t n, m;
    std::cin >> n >> m;
    matrix<double> u(n, m);
    random_filling(u, 10.0, 9876);
    random_bounds(u, 100, 5.0, 9876);

    double t0 = omp_get_wtime();

    laplace2d(u, 1.0 / (n-1), 1.0 / (n-1));

    double t1 = omp_get_wtime();

    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
              << "Timing: " << t1 - t0 << " sec\n"
              << "Laplace equation satisfied: " << test_laplace(u, 1e-6)
              << std::endl;
    return 0;
}
