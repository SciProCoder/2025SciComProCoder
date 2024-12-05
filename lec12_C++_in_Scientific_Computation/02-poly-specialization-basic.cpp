#include "./include/polymorphism_example.h"

namespace Example1 {

template <typename T>
void AXPY(const int n, const T a, const T *x, const int incx, T *y, const int incy)
{
    std::cout << "Generic AXPY" << std::endl;
    for (int i = 0; i < n; ++i)
    {
        y[i * incy] += a * x[i * incx];
    }
}
template <>
void AXPY(const int n, const double a, const double *x, const int incx, double *y, const int incy)
{
    std::cout << "Specialized AXPY" << std::endl;
    cblas_daxpy(n, a, x, incx, y, incy);
}
}; // namespace Example1

int main()
{
    {
        using namespace Example1;

        std::cout << "Case 1: " << std::endl;

        std::vector<double> x     = {1, 2, 3, 4, 5};
        std::vector<double> y     = {6, 7, 8, 9, 10};
        double              alpha = 2.0;

        AXPY(x.size(), alpha, x.data(), 1, y.data(), 1);
    }

    {
        using namespace Example1; // template 2

        std::cout << "Case 2: " << std::endl;

        std::vector<complex_double> x     = {1, 2, 3, 4, 5};
        std::vector<complex_double> y     = {6, 7, 8, 9, 10};
        complex_double              alpha = 2.0;

        AXPY(x.size(), alpha, x.data(), 1, y.data(), 1);
    }
}