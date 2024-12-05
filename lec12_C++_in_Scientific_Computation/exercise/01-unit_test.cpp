#include <iostream>
#include <cassert>
#include <complex>
#include <vector>
#include "../include/basic.h"
#include "01-linear_algebra_interface.h"

using namespace Training2024::LinearAlgebraInterface;

// 测试 REAL 和 IMAG 函数
void TestRealAndImag()
{
    std::cout << "Testing REAL and IMAG functions...\n";

    assert(REAL(3.14f) == 3.14f);
    assert(REAL(3.14) == 3.14);
    assert(REAL(std::complex<float>(2.0f, 3.0f)) == 2.0f);
    assert(REAL(std::complex<double>(2.0, 3.0)) == 2.0);

    assert(IMAG(0.0f) == 0.0f);
    assert(IMAG(0.0) == 0.0);
    assert(IMAG(std::complex<float>(2.0f, 3.0f)) == 3.0f);
    assert(IMAG(std::complex<double>(2.0, 3.0)) == 3.0);

    std::cout << "REAL and IMAG tests passed.\n";
}

// 测试 NORM 和 NORM_SQUARE
void TestNormAndNormSquare()
{
    std::cout << "Testing NORM and NORM_SQUARE functions...\n";

    assert(NORM(3.0f) == 3.0f);
    assert(NORM(3.0) == 3.0);
    assert(NORM(std::complex<float>(3.0f, 4.0f)) == 5.0f);
    assert(NORM(std::complex<double>(3.0, 4.0)) == 5.0);

    assert(NORM_SQUARE(3.0f) == 9.0f);
    assert(NORM_SQUARE(3.0) == 9.0);
    assert(NORM_SQUARE(std::complex<float>(3.0f, 4.0f)) == 25.0f);
    assert(NORM_SQUARE(std::complex<double>(3.0, 4.0)) == 25.0);

    std::cout << "NORM and NORM_SQUARE tests passed.\n";
}

// 测试矩阵打印功能
void TestPrintMatrix()
{
    std::cout << "Testing PRINT_MATRIX function...\n";

    float matrix[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::cout << "Expected output:\n1 2 \n3 4\n";
    std::cout << "Actual output:\n";
    PRINT_MATRIX(matrix, 2, 2); // 应打印2x2矩阵
    std::cout << "Check the printed output for correctness.\n";
}

// 测试 SCAL 和 AXPY 函数
void TestSCALAndAXPY()
{
    std::cout << "Testing SCAL and AXPY functions...\n";

    float x[] = {1.0f, 2.0f, 3.0f};
    SCAL(3, 2.0f, x, 1);
    assert(x[0] == 2.0f);
    assert(x[1] == 4.0f);
    assert(x[2] == 6.0f);

    float y[] = {4.0f, 5.0f, 6.0f};
    AXPY(3, 2.0f, x, 1, y, 1);
    assert(y[0] == 8.0f);
    assert(y[1] == 13.0f);
    assert(y[2] == 18.0f);

    std::cout << "SCAL and AXPY tests passed.\n";
}

// 测试 MATRIXMUL 函数
void TestMatrixMul()
{
    std::cout << "Testing MATRIXMUL function...\n";

    float A[]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[]  = {1.0f, 0.0f, 0.0f, 1.0f};
    float C[4] = {0.0f};

    MATRIXMUL(2, 2, 2, A, B, C);
    assert(C[0] == 1.0f);
    assert(C[1] == 2.0f);
    assert(C[2] == 3.0f);
    assert(C[3] == 4.0f);

    std::cout << "MATRIXMUL test passed.\n";
}

// 测试 Hermitian 矩阵对角化
void TestHermitianMatDiag()
{
    std::cout << "Testing Hermitian Matrix Diagonalization...\n";

    std::vector<lapack_complex_double> matrix = {{4.0, 0.0}, {1.0, -1.0}, {0.0, -2.0}, {2.0, 1.0},  {3.0, -1.0}, {1.0, 1.0},  {3.0, 0.0},  {1.0, -1.0}, {0.0, 2.0},
                                                 {2.0, 1.0}, {0.0, 2.0},  {1.0, 1.0},  {5.0, 0.0},  {1.0, -2.0}, {0.0, -1.0}, {2.0, -1.0}, {0.0, -2.0}, {1.0, 2.0},
                                                 {2.0, 0.0}, {1.0, -1.0}, {3.0, 1.0},  {2.0, -1.0}, {0.0, 1.0},  {1.0, 1.0},  {6.0, 0.0}};

    std::vector<double>                eigenvalues(5);
    std::vector<lapack_complex_double> eigenvectors(25);

    HermMatDiag(LAPACK_ROW_MAJOR, 'V', 'U', 5, matrix.data(), 5, eigenvalues.data());

    std::cout << "Eigenvalues:\n";
    for (double val : eigenvalues)
    {
        std::cout << val << " ";
    }
    std::cout << "\n";

    assert(eigenvalues[0] <= eigenvalues[1]);
    assert(eigenvalues[1] <= eigenvalues[2]);
    assert(eigenvalues[2] <= eigenvalues[3]);
    assert(eigenvalues[3] <= eigenvalues[4]);

    std::cout << "Hermitian Matrix Diagonalization test passed.\n";
}

int main()
{
    TestRealAndImag();
    TestNormAndNormSquare();
    TestPrintMatrix();
    TestSCALAndAXPY();
    TestMatrixMul();
    TestHermitianMatDiag();

    std::cout << "All tests passed.\n";
    return 0;
}
