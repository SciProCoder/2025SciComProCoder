#pragma once

#include "../include/basic.h"

namespace Training2024 {
namespace LinearAlgebraInterface {

/* ---------- function overload for BLAS ---------- */

/* ------------------- */

// return the real part of a scalar

inline float  REAL(const float X);
inline double REAL(const double X);
inline float  REAL(const complex_float X);
inline double REAL(const complex_double X);

// return the imaginary part of a scalar

inline float  IMAG(const float X);
inline double IMAG(const double X);
inline float  IMAG(const complex_float X);
inline double IMAG(const complex_double X);

/* ------------------- */

/* NORM, |X| */

inline float  NORM(const float X);
inline double NORM(const double X);
inline float  NORM(const complex_float X);
inline double NORM(const complex_double X);

/* NORM_SQUARE, |X|^2 */

inline float  NORM_SQUARE(const float X);
inline double NORM_SQUARE(const double X);
inline float  NORM_SQUARE(const complex_float X);
inline double NORM_SQUARE(const complex_double X);

/* PRINT */

template <std::floating_point Scalar>
void PRINT_SCALAR(const Scalar);

/* PRINT_MATRIX */

template <std::floating_point Scalar>
void PRINT_MATRIX(const Scalar *_Input, const int nRow, const int nCol);

/* COMPLEX CONJUGATE */

inline float          CONJ(const float X);
inline double         CONJ(const double X);
inline complex_float  CONJ(const complex_float X);
inline complex_double CONJ(const complex_double X);
inline void           CONJ(const int N, double *X);
inline void           CONJ(const int N, complex_double *X);

/* HERMITIAN CONJUGATE */

template <std::floating_point Scalar>
std::vector<Scalar> HERMITIAN_CONJ(std::vector<Scalar> &_Input, const int nRow, const int nCol);

/* ----- BLAS 1 ----- */

/* SCAL: x = ax */

inline void SCAL(const MKL_INT n, const float a, float *x, const MKL_INT incx);
inline void SCAL(const MKL_INT n, const double a, double *x, const MKL_INT incx);
inline void SCAL(const MKL_INT n, const complex_float a, complex_float *x, const MKL_INT incx);
inline void SCAL(const MKL_INT n, const complex_double a, complex_double *x, const MKL_INT incx);
inline void SCAL(const MKL_INT n, const float a, complex_float *x, const MKL_INT incx);
inline void SCAL(const MKL_INT n, const double a, complex_double *x, const MKL_INT incx);

/* AXPY: y = ax + y */

inline void AXPY(const MKL_INT n, const float a, const float *x, const MKL_INT incx, float *y, const MKL_INT incy);
inline void AXPY(const MKL_INT n, const double a, const double *x, const MKL_INT incx, double *y, const MKL_INT incy);
inline void AXPY(const MKL_INT n, const complex_float a, const complex_float *x, const MKL_INT incx, complex_float *y, const MKL_INT incy);
inline void AXPY(const MKL_INT n, const complex_double a, const complex_double *x, const MKL_INT incx, complex_double *y, const MKL_INT incy);

/* COPY : y = x */

inline void COPY(const MKL_INT n, const float *x, const MKL_INT incx, float *y, const MKL_INT incy);
inline void COPY(const MKL_INT n, const double *x, const MKL_INT incx, double *y, const MKL_INT incy);
inline void COPY(const MKL_INT n, const complex_float *x, const MKL_INT incx, complex_float *y, const MKL_INT incy);
inline void COPY(const MKL_INT n, const complex_double *x, const MKL_INT incx, complex_double *y, const MKL_INT incy);

/* Inner Product: (x,y) */

inline float          DOT(const MKL_INT n, const float *x, const MKL_INT incx, const float *y, const MKL_INT incy);
inline double         DOT(const MKL_INT n, const double *x, const MKL_INT incx, const double *y, const MKL_INT incy);
inline complex_float  DOT(const MKL_INT n, const complex_float *x, const MKL_INT incx, const complex_float *y, const MKL_INT incy);
inline complex_double DOT(const MKL_INT n, const complex_double *x, const MKL_INT incx, const complex_double *y, const MKL_INT incy);

/* Norm of vector */

inline float  NORM2_VEC(const MKL_INT n, const float *x, const MKL_INT incx);
inline double NORM2_VEC(const MKL_INT n, const double *x, const MKL_INT incx);
inline float  NORM2_VEC(const MKL_INT n, const complex_float *x, const MKL_INT incx);
inline double NORM2_VEC(const MKL_INT n, const complex_double *x, const MKL_INT incx);

inline float  NORM2(const MKL_INT n, const float *x, const MKL_INT incx);
inline double NORM2(const MKL_INT n, const double *x, const MKL_INT incx);
inline float  NORM2(const MKL_INT n, const complex_float *x, const MKL_INT incx);
inline double NORM2(const MKL_INT n, const complex_double *x, const MKL_INT incx);

/* ----- BLAS 2 ----- */

/* Rank 1 update, A = alpha * outer(x,y) + A */

inline void GER(const MKL_INT M, const MKL_INT N, const float alpha, const float *X, const MKL_INT incX, const float *Y, const MKL_INT incY, float *A, const MKL_INT lda);
inline void GER(const MKL_INT M, const MKL_INT N, const double alpha, const double *X, const MKL_INT incX, const double *Y, const MKL_INT incY, double *A, const MKL_INT lda);
inline void GER(const MKL_INT M, const MKL_INT N, const complex_float alpha, const complex_float *X, const MKL_INT incX, const complex_float *Y, const MKL_INT incY, complex_float *A,
                const MKL_INT lda);
inline void GER(const MKL_INT M, const MKL_INT N, const complex_double alpha, const complex_double *X, const MKL_INT incX, const complex_double *Y, const MKL_INT incY, complex_double *A,
                const MKL_INT lda);

/* ----- BLAS 3 ----- */

/* Matrix Product, no transpose, NOTE A is of size M * K, B is of size K * N */

inline void MATRIXMUL(const int M, const int K, const int N, const float alpha, const float beta, const float *A, const float *B, float *C);
inline void MATRIXMUL(const int M, const int K, const int N, const double alpha, const double beta, const double *A, const double *B, double *C);
inline void MATRIXMUL(const int M, const int K, const int N, const complex_float alpha, const complex_float beta, const complex_float *A, const complex_float *B, complex_float *C);
inline void MATRIXMUL(const int M, const int K, const int N, const complex_double alpha, const complex_double beta, const complex_double *A, const complex_double *B, complex_double *C);

/* Matrix Product, no transpose, NOTE A is of size M * K, B is of size K * N, alpha = 1, beta = 0 */

inline void MATRIXMUL(const int M, const int K, const int N, const float *A, const float *B, float *C);
inline void MATRIXMUL(const int M, const int K, const int N, const double *A, const double *B, double *C);
inline void MATRIXMUL(const int M, const int K, const int N, const complex_float *A, const complex_float *B, complex_float *C);
inline void MATRIXMUL(const int M, const int K, const int N, const complex_double *A, const complex_double *B, complex_double *C);

inline void GEMM(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N, const MKL_INT K, const float alpha, const float *A,
                 const MKL_INT lda, const float *B, const MKL_INT ldb, const float beta, float *C, const MKL_INT ldc);
inline void GEMM(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N, const MKL_INT K, const double alpha, const double *A,
                 const MKL_INT lda, const double *B, const MKL_INT ldb, const double beta, double *C, const MKL_INT ldc);
inline void GEMM(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N, const MKL_INT K, const complex_float alpha,
                 const complex_float *A, const MKL_INT lda, const complex_float *B, const MKL_INT ldb, const complex_float beta, complex_float *C, const MKL_INT ldc);
inline void GEMM(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N, const MKL_INT K, const complex_double alpha,
                 const complex_double *A, const MKL_INT lda, const complex_double *B, const MKL_INT ldb, const complex_double beta, complex_double *C, const MKL_INT ldc);

/* lapacke */

/* ----- Hermitian Matrix Diagonalization ----- */

inline lapack_int HermMatDiag(int matrix_layout, char jobz, char uplo, lapack_int n, float *a, lapack_int lda, float *w);
inline lapack_int HermMatDiag(int matrix_layout, char jobz, char uplo, lapack_int n, double *a, lapack_int lda, double *w);
inline lapack_int HermMatDiag(int matrix_layout, char jobz, char uplo, lapack_int n, lapack_complex_float *a, lapack_int lda, float *w);
inline lapack_int HermMatDiag(int matrix_layout, char jobz, char uplo, lapack_int n, lapack_complex_double *a, lapack_int lda, double *w);

/* **************************************************************************************** */
}; // namespace LinearAlgebraInterface
}; // namespace Training2024