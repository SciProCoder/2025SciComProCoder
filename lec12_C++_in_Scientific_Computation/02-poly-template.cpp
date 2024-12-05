#include "./include/polymorphism_example.h"

namespace Example1 {
inline void AXPY(const MKL_INT n, const double a, const double *x, const MKL_INT incx, double *y, const MKL_INT incy)
{
    std::cout << "Function AXPY" << std::endl;
    cblas_daxpy(n, a, x, incx, y, incy);
}

template <typename T>
    requires(!std::is_same_v<T, double>)
void AXPY(const int n, const T a, const T *x, const int incx, T *y, const int incy)
{
    std::cout << "Template AXPY of type 1" << std::endl;
    for (int i = 0; i < n; ++i)
    {
        y[i * incy] += a * x[i * incx];
    }
}

template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>)
void AXPY(const int n, const T1 a, const T2 *x, const int incx, T2 *y, const int incy)
{
    std::cout << "Template AXPY of type 2" << std::endl;
    for (int i = 0; i < n; ++i)
    {
        y[i * incy] += a * x[i * incx];
    }
}

}; // namespace Example1

namespace Example2 {
template <typename T>
void AXPY(const int n, const T a, const T *x, const int incx, T *y, const int incy)
{
    std::cout << "Template AXPY of type 1" << std::endl;
    for (int i = 0; i < n; ++i)
    {
        y[i * incy] += a * x[i * incx];
    }
}
template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>)
void AXPY(const int n, const T1 a, const T2 *x, const int incx, T2 *y, const int incy)
{
    std::cout << "Template AXPY of type 2" << std::endl;
    for (int i = 0; i < n; ++i)
    {
        y[i * incy] += a * x[i * incx];
    }
}
}; // namespace Example2

template <typename... Args>
void print(Args... args)
{
    ((std::cout << args << " "), ...) << std::endl; // C++17 折叠表达式
}

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
        using namespace Example2; // template 2

        std::cout << "Case 2: " << std::endl;

        std::vector<double> x     = {1, 2, 3, 4, 5};
        std::vector<double> y     = {6, 7, 8, 9, 10};
        double              alpha = 2.0;

        AXPY(x.size(), alpha, x.data(), 1, y.data(), 1);
    }

    {
        using namespace Example1;

        std::cout << "Case 3: " << std::endl;

        std::vector<complex_double> x     = {1, 2, 3, 4, 5};
        std::vector<complex_double> y     = {6, 7, 8, 9, 10};
        double                      alpha = 2.0;

        AXPY(x.size(), alpha, x.data(), 1, y.data(), 1);
    }

    {
        print(1, 2, 3);
        print("hello", "world");
        print(2024, "SciComProCoder");
    }
}