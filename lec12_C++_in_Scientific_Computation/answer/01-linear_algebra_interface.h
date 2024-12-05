#pragma once

#include "../include/basic.h"

namespace Training2024 {
namespace LinearAlgebraInterface {

/* ---------- function overload for BLAS ---------- */

/* ------------------- */

// return the real part of a scalar

inline float  REAL(const float X) { return X; }
inline double REAL(const double X) { return X; }
inline float  REAL(const complex_float X) { return X.real(); }
inline double REAL(const complex_double X) { return X.real(); }

// return the imaginary part of a scalar

inline float  IMAG([[maybe_unused]] const float X) { return 0.0; }
inline double IMAG([[maybe_unused]] const double X) { return 0.0; }
inline float  IMAG(const complex_float X) { return X.imag(); }
inline double IMAG(const complex_double X) { return X.imag(); }

/* ------------------- */

/* NORM, |X| */

inline float  NORM(const float X) { return fabs(X); }
inline double NORM(const double X) { return fabs(X); }
inline float  NORM(const complex_float X) { return std::fabs(X); }
inline double NORM(const complex_double X) { return std::fabs(X); }

/* NORM_SQUARE, |X|^2 */

inline float  NORM_SQUARE(const float X) { return X * X; }
inline double NORM_SQUARE(const double X) { return X * X; }
inline float  NORM_SQUARE(const complex_float X) { return X.real() * X.real() + X.imag() * X.imag(); }
inline double NORM_SQUARE(const complex_double X) { return X.real() * X.real() + X.imag() * X.imag(); }

/* PRINT */

template <std::floating_point Scalar>
void PRINT_SCALAR(const Scalar);

template <>
inline void PRINT_SCALAR<double>(const double _input)
{
    printf("%15.8f ", _input);
}
template <>
inline void PRINT_SCALAR<float>(const float _input)
{
    printf("%15.8f ", _input);
}
template <>
inline void PRINT_SCALAR<complex_float>(const complex_float _input)
{
    printf("%15.8f + %15.8f ", _input.real(), _input.imag());
}
template <>
inline void PRINT_SCALAR<complex_double>(const complex_double _input)
{
    printf("%15.8f + %15.8f ", _input.real(), _input.imag());
}

/* PRINT_MATRIX */

template <std::floating_point Scalar>
void PRINT_MATRIX(const Scalar *_Input, const int nRow, const int nCol)
{
    printf("--------------------------------------------------\n");
    for (int irow = 0; irow < nRow; ++irow)
    {
        for (int icol = 0; icol < nCol; ++icol)
        {
            PRINT_SCALAR<Scalar>(_Input[irow * nCol + icol]);
        }
        printf("\n");
    }
    printf("--------------------------------------------------\n");
}

/* COMPLEX CONJUGATE */

inline float          CONJ(const float X) { return X; }
inline double         CONJ(const double X) { return X; }
inline complex_float  CONJ(const complex_float X) { return std::conj(X); }
inline complex_double CONJ(const complex_double X) { return std::conj(X); }

/* HERMITIAN CONJUGATE */

template <std::floating_point Scalar>
std::vector<Scalar> HERMITIAN_CONJ(std::vector<Scalar> &_Input, const int nRow, const int nCol)
{
    {
        std::vector<Scalar> Res(nCol * nRow);
        for (int irow = 0; irow < nRow; ++irow)
        {
            for (int icol = 0; icol < nCol; ++icol)
            {
                Res[icol * nRow + irow] = CONJ(_Input[irow * nCol + icol]);
            }
        }
        return Res;
    }
}

/* ----- BLAS 1 ----- */

/* SCAL: x = ax */

inline void SCAL(const MKL_INT n, const float a, float *x, const MKL_INT incx) { cblas_sscal(n, a, x, incx); }
inline void SCAL(const MKL_INT n, const double a, double *x, const MKL_INT incx) { cblas_dscal(n, a, x, incx); }
inline void SCAL(const MKL_INT n, const complex_float a, complex_float *x, const MKL_INT incx) { cblas_cscal(n, &a, x, incx); }
inline void SCAL(const MKL_INT n, const complex_double a, complex_double *x, const MKL_INT incx) { cblas_zscal(n, &a, x, incx); }
inline void SCAL(const MKL_INT n, const float a, complex_float *x, const MKL_INT incx) { cblas_csscal(n, a, x, incx); }
inline void SCAL(const MKL_INT n, const double a, complex_double *x, const MKL_INT incx) { cblas_zdscal(n, a, x, incx); }

/* AXPY: y = ax + y */

inline void AXPY(const MKL_INT n, const float a, const float *x, const MKL_INT incx, float *y, const MKL_INT incy) { cblas_saxpy(n, a, x, incx, y, incy); }
inline void AXPY(const MKL_INT n, const double a, const double *x, const MKL_INT incx, double *y, const MKL_INT incy) { cblas_daxpy(n, a, x, incx, y, incy); }
inline void AXPY(const MKL_INT n, const complex_float a, const complex_float *x, const MKL_INT incx, complex_float *y, const MKL_INT incy)
{
    cblas_caxpy(n, (const void *)&a, x, incx, y, incy);
}
inline void AXPY(const MKL_INT n, const complex_double a, const complex_double *x, const MKL_INT incx, complex_double *y, const MKL_INT incy)
{
    cblas_zaxpy(n, (const void *)&a, x, incx, y, incy);
}
/* COPY : y = x */

inline void COPY(const MKL_INT n, const float *x, const MKL_INT incx, float *y, const MKL_INT incy) { cblas_scopy(n, x, incx, y, incy); }
inline void COPY(const MKL_INT n, const double *x, const MKL_INT incx, double *y, const MKL_INT incy) { cblas_dcopy(n, x, incx, y, incy); }
inline void COPY(const MKL_INT n, const complex_float *x, const MKL_INT incx, complex_float *y, const MKL_INT incy) { cblas_ccopy(n, x, incx, y, incy); }
inline void COPY(const MKL_INT n, const complex_double *x, const MKL_INT incx, complex_double *y, const MKL_INT incy) { cblas_zcopy(n, x, incx, y, incy); }

/* Inner Product: (x,y) */

inline float         DOT(const MKL_INT n, const float *x, const MKL_INT incx, const float *y, const MKL_INT incy) { return cblas_sdot(n, x, incx, y, incy); }
inline double        DOT(const MKL_INT n, const double *x, const MKL_INT incx, const double *y, const MKL_INT incy) { return cblas_ddot(n, x, incx, y, incy); }
inline complex_float DOT(const MKL_INT n, const complex_float *x, const MKL_INT incx, const complex_float *y, const MKL_INT incy)
{
    complex_float Res{0.0, 0.0};
    cblas_cdotc_sub(n, x, incx, y, incy, &Res);
    return Res;
}
inline complex_double DOT(const MKL_INT n, const complex_double *x, const MKL_INT incx, const complex_double *y, const MKL_INT incy)
{
    complex_double Res{0.0, 0.0};
    cblas_zdotc_sub(n, x, incx, y, incy, &Res);
    return Res;
}

/* Norm of vector */

inline float  NORM2_VEC(const MKL_INT n, const float *x, const MKL_INT incx) { return cblas_snrm2(n, x, incx); }
inline double NORM2_VEC(const MKL_INT n, const double *x, const MKL_INT incx) { return cblas_dnrm2(n, x, incx); }
inline float  NORM2_VEC(const MKL_INT n, const complex_float *x, const MKL_INT incx) { return cblas_scnrm2(n, x, incx); }
inline double NORM2_VEC(const MKL_INT n, const complex_double *x, const MKL_INT incx) { return cblas_dznrm2(n, x, incx); }

inline float  NORM2(const MKL_INT n, const float *x, const MKL_INT incx) { return NORM2_VEC(n, x, incx); }
inline double NORM2(const MKL_INT n, const double *x, const MKL_INT incx) { return NORM2_VEC(n, x, incx); }
inline float  NORM2(const MKL_INT n, const complex_float *x, const MKL_INT incx) { return NORM2_VEC(n, x, incx); }
inline double NORM2(const MKL_INT n, const complex_double *x, const MKL_INT incx) { return NORM2_VEC(n, x, incx); }

/* ----- BLAS 2 ----- */

/* Rank 1 update, A = alpha * outer(x,y) + A */

inline void GER(const MKL_INT M, const MKL_INT N, const float alpha, const float *X, const MKL_INT incX, const float *Y, const MKL_INT incY, float *A, const MKL_INT lda)
{
    cblas_sger(CblasRowMajor, M, N, alpha, X, incX, Y, incY, A, lda);
}
inline void GER(const MKL_INT M, const MKL_INT N, const double alpha, const double *X, const MKL_INT incX, const double *Y, const MKL_INT incY, double *A, const MKL_INT lda)
{
    cblas_dger(CblasRowMajor, M, N, alpha, X, incX, Y, incY, A, lda);
}
inline void GER(const MKL_INT M, const MKL_INT N, const complex_float alpha, const complex_float *X, const MKL_INT incX, const complex_float *Y, const MKL_INT incY, complex_float *A,
                const MKL_INT lda)
{
    cblas_cgerc(CblasRowMajor, M, N, &alpha, X, incX, Y, incY, A, lda);
}
inline void GER(const MKL_INT M, const MKL_INT N, const complex_double alpha, const complex_double *X, const MKL_INT incX, const complex_double *Y, const MKL_INT incY, complex_double *A,
                const MKL_INT lda)
{
    cblas_zgerc(CblasRowMajor, M, N, &alpha, X, incX, Y, incY, A, lda);
}

/* ----- BLAS 3 ----- */

/* Matrix Product, no transpose, NOTE A is of size M * K, B is of size K * N */

inline void MATRIXMUL(const int M, const int K, const int N, const float alpha, const float beta, const float *A, const float *B, float *C)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
}
inline void MATRIXMUL(const int M, const int K, const int N, const double alpha, const double beta, const double *A, const double *B, double *C)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
}
inline void MATRIXMUL(const int M, const int K, const int N, const complex_float alpha, const complex_float beta, const complex_float *A, const complex_float *B, complex_float *C)
{
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, &alpha, A, K, B, N, &beta, C, N);
}
inline void MATRIXMUL(const int M, const int K, const int N, const complex_double alpha, const complex_double beta, const complex_double *A, const complex_double *B, complex_double *C)
{
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, &alpha, A, K, B, N, &beta, C, N);
}

/* Matrix Product, no transpose, NOTE A is of size M * K, B is of size K * N, alpha = 1, beta = 0 */

inline void MATRIXMUL(const int M, const int K, const int N, const float *A, const float *B, float *C) { MATRIXMUL(M, K, N, 1.0, 0.0, A, B, C); }
inline void MATRIXMUL(const int M, const int K, const int N, const double *A, const double *B, double *C) { MATRIXMUL(M, K, N, 1.0, 0.0, A, B, C); }
inline void MATRIXMUL(const int M, const int K, const int N, const complex_float *A, const complex_float *B, complex_float *C)
{
    static const complex_float alpha(1.0, 0.0), beta(0.0, 0.0);
    MATRIXMUL(M, K, N, alpha, beta, A, B, C);
}
inline void MATRIXMUL(const int M, const int K, const int N, const complex_double *A, const complex_double *B, complex_double *C)
{
    static const complex_double alpha(1.0, 0.0), beta(0.0, 0.0);
    MATRIXMUL(M, K, N, alpha, beta, A, B, C);
}

inline void GEMM(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, //
                 const MKL_INT M, const MKL_INT N, const MKL_INT K,                                     //
                 const float  alpha,                                                                    //
                 const float *A, const MKL_INT lda,                                                     //
                 const float *B, const MKL_INT ldb,                                                     //
                 const float beta,                                                                      //
                 float *C, const MKL_INT ldc)                                                           //
{
    cblas_sgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline void GEMM(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, //
                 const MKL_INT M, const MKL_INT N, const MKL_INT K,                                     //
                 const double  alpha,                                                                   //
                 const double *A, const MKL_INT lda,                                                    //
                 const double *B, const MKL_INT ldb,                                                    //
                 const double beta,                                                                     //
                 double *C, const MKL_INT ldc)                                                          //
{
    cblas_dgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline void GEMM(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, //
                 const MKL_INT M, const MKL_INT N, const MKL_INT K,                                     //
                 const complex_float  alpha,                                                            //
                 const complex_float *A, const MKL_INT lda,                                             //
                 const complex_float *B, const MKL_INT ldb,                                             //
                 const complex_float beta,                                                              //
                 complex_float *C, const MKL_INT ldc)                                                   //
{
    cblas_cgemm(Layout, TransA, TransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}
inline void GEMM(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, //
                 const MKL_INT M, const MKL_INT N, const MKL_INT K,                                     //
                 const complex_double  alpha,                                                           //
                 const complex_double *A, const MKL_INT lda,                                            //
                 const complex_double *B, const MKL_INT ldb,                                            //
                 const complex_double beta,                                                             //
                 complex_double *C, const MKL_INT ldc)                                                  //
{
    cblas_zgemm(Layout, TransA, TransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

/* lapacke */

/* ----- Hermitian Matrix Diagonalization ----- */

inline lapack_int HermMatDiag(int matrix_layout, char jobz, char uplo, lapack_int n, float *a, lapack_int lda, float *w) { return LAPACKE_ssyevd(matrix_layout, jobz, uplo, n, a, lda, w); }
inline lapack_int HermMatDiag(int matrix_layout, char jobz, char uplo, lapack_int n, double *a, lapack_int lda, double *w) { LAPACKE_dsyevd(matrix_layout, jobz, uplo, n, a, lda, w); }
inline lapack_int HermMatDiag(int matrix_layout, char jobz, char uplo, lapack_int n, lapack_complex_float *a, lapack_int lda, float *w)
{
    return LAPACKE_cheevd(matrix_layout, jobz, uplo, n, a, lda, w);
}
inline lapack_int HermMatDiag(int matrix_layout, char jobz, char uplo, lapack_int n, lapack_complex_double *a, lapack_int lda, double *w)
{
    return LAPACKE_zheevd(matrix_layout, jobz, uplo, n, a, lda, w);
}

}; // namespace LinearAlgebraInterface
}; // namespace Training2024