#include "./include/polymorphism_example.h"

int main()
{
    {
        using namespace Training2024::poly_example03_1;

        std::vector<double> x     = {1, 2, 3, 4, 5};
        std::vector<double> y     = {6, 7, 8, 9, 10};
        double              alpha = 2.0;

        automatic_unroll<double, 5>::AXPY(alpha, x.data(), y.data());

        for (size_t i = 0; i < x.size(); ++i)
        {
            std::cout << y[i] << " ";
        }

        std::cout << std::endl;
    }
}
