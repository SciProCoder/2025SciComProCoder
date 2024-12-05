#include "./include/polymorphism_example.h"

int main()
{
    {
        using namespace Training2024::poly_example01;

        std::vector<double> x = {1, 2, 3, 4, 5};
        std::vector<double> y = {6, 7, 8, 9, 10};
        double alpha = 2.0;

        std::vector<complex_double> cx = {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}};
        std::vector<complex_double> cy = {{11, 12}, {13, 14}, {15, 16}, {17, 18}, {19, 20}};

        // 同样的函数名可用于不同类型的输入参数
        AXPY(x.size(), alpha, x.data(), 1, y.data(), 1);
        AXPY(cx.size(), complex_double(2, 0), cx.data(), 1, cy.data(), 1);

        for (size_t i = 0; i < x.size(); ++i)
        {
            std::cout << y[i] << " ";
        }

        std::cout << std::endl;

        for (size_t i = 0; i < cx.size(); ++i)
        {
            std::cout << cy[i] << " ";
        }

        std::cout << std::endl;
    }

    {
        using namespace Training2024::poly_example02;

        std::vector<double> x = {1, 2, 3, 4, 5};
        std::vector<double> y = {6, 7, 8, 9, 10};
        VecView<double> vx(x.data(), x.size());
        VecView<double> vy(y.data(), y.size());

        double alpha = 2.0;

        std::vector<complex_double> cx = {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}};
        std::vector<complex_double> cy = {{11, 12}, {13, 14}, {15, 16}, {17, 18}, {19, 20}};

        VecView<complex_double> vcx(cx.data(), cx.size());
        VecView<complex_double> vcy(cy.data(), cy.size());

        // 更加简洁的调用方式
        AXPY(alpha, vx, vy);
        AXPY(complex_double(2, 0), vcx, vcy);

        for (size_t i = 0; i < x.size(); ++i)
        {
            std::cout << y[i] << " ";
        }

        std::cout << std::endl;

        for (size_t i = 0; i < cx.size(); ++i)
        {
            std::cout << cy[i] << " ";
        }

        std::cout << std::endl;
    }
}