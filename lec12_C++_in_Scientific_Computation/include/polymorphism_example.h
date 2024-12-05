#pragma once

#include "basic.h"

#include <concepts>

namespace Training2024 {
namespace poly_example01 {
inline void AXPY(const MKL_INT n, const float a, const float *x, const MKL_INT incx, float *y, const MKL_INT incy) { cblas_saxpy(n, a, x, incx, y, incy); }
inline void AXPY(const MKL_INT n, const double a, const double *x, const MKL_INT incx, double *y, const MKL_INT incy) { cblas_daxpy(n, a, x, incx, y, incy); }
inline void AXPY(const MKL_INT n, const complex_float a, const complex_float *x, const MKL_INT incx, complex_float *y, const MKL_INT incy) { cblas_caxpy(n, &a, x, incx, y, incy); }
inline void AXPY(const MKL_INT n, const complex_double a, const complex_double *x, const MKL_INT incx, complex_double *y, const MKL_INT incy) { cblas_zaxpy(n, &a, x, incx, y, incy); }
}; // namespace poly_example01

namespace poly_example02 {
template <::std::floating_point T>
class VecView
{
public:
    VecView(T *data, size_t n, size_t stride = 1) : data_(data), n_(n), stride_(stride) {}

    T       &operator[](size_t i) { return data_[i * stride_]; }
    const T &operator[](size_t i) const { return data_[i * stride_]; }

    size_t size() const { return n_; }

    // protected:
    T     *data_;
    size_t n_;
    size_t stride_;
};

template <std::floating_point T>
inline void AXPY(const T a, const VecView<T> &x, VecView<T> &y)
{
    Training2024::poly_example01::AXPY(x.size(), a, x.data_, x.stride_, y.data_, y.stride_);
}

template <std::floating_point T>
inline void AXPY_PlainCPP(const MKL_INT n, const T a, const T *x, const MKL_INT incx, T *y, const MKL_INT incy)
{
    for (MKL_INT i = 0; i < n; ++i)
    {
        y[i * incy] += a * x[i * incx];
    }
}

}; // namespace poly_example02

namespace poly_example03_1 {
template <std::floating_point Scalar, int N>
class automatic_unroll
{
public:
    static void AXPY(const Scalar a, const Scalar *x, Scalar *y)
    {
        *y += a * *x;
        automatic_unroll<Scalar, N - 1>::AXPY(a, x + 1, y + 1);
    }
};

template <std::floating_point Scalar>
class automatic_unroll<Scalar, 1>
{
public:
    static void AXPY(const Scalar a, const Scalar *x, Scalar *y) { *y += a * *x; }
};
}; // namespace poly_example03_1

namespace poly_example03_2 {
template <int M, int N>
    requires(M >= 0 && N >= 0)
class combination
{
public:
    static constexpr size_t value = (M < N) ? 0 : combination<M - 1, N - 1>::value + combination<M - 1, N>::value;
};

template <int M>
class combination<M, 0>
{
public:
    static constexpr size_t value = 1;
};

template <int M>
class combination<M, M>
{
public:
    static constexpr size_t value = 1;
};
}; // namespace poly_example03_2

namespace poly_example04 {
inline double get_wtime() { return omp_get_wtime(); }

inline double get_cputime() { return (double)clock() / (double)CLOCKS_PER_SEC; }

class OpenMPAlgoBase
{
public:
    /* constructor and destructor */

    OpenMPAlgoBase() noexcept : num_of_threads{OMP_NUM_OF_THREADS} {}
    OpenMPAlgoBase(int _num_of_threads) noexcept : num_of_threads{_num_of_threads} {}
    virtual ~OpenMPAlgoBase() noexcept = default;

    /* setter and getter */

    SETTER(size_t, begin_indx, OpenMPAlgoBase)
    SETTER(size_t, end_indx, OpenMPAlgoBase)
    GETTER(size_t, begin_indx)
    GETTER(size_t, end_indx)
    GETTER(int, num_of_threads)

    /* function -- interface */

    DECLARE_VIRTUAL_FUNCTION(initialize)
    DECLARE_VIRTUAL_FUNCTION(finish)
    DECLARE_PURE_VIRTUAL_FUNCTION(initialize_local)
    DECLARE_PURE_VIRTUAL_FUNCTION(finish_local)
    virtual void do_task(size_t _task_id) = 0;
    virtual bool should_upload_data() noexcept { return true; }
    virtual bool should_upload_data_force() noexcept { return true; }
    DECLARE_PURE_VIRTUAL_FUNCTION(upload_data)
    DECLARE_PURE_VIRTUAL_FUNCTION(upload_data_force)
    DECLARE_VIRTUAL_FUNCTION(get_time_begin)
    DECLARE_VIRTUAL_FUNCTION(get_time_end)

    /* RUN */

    DECLARE_VIRTUAL_FUNCTION(run)

    /* time module */
    double get_cputime_tot() noexcept { return cpu_end - cpu_begin; }
    double get_cputime_interval() noexcept { return cpu2 - cpu1; }
    double get_walltime_tot() noexcept { return wall_end - wall_begin; }
    double get_walltime_interval() noexcept { return wall2 - wall1; }

protected:
    int    num_of_threads = 0;
    size_t begin_indx = 0, end_indx = 0; /// NOTE: int 的范围可能不够.
    /* 计时模块 */
    double cpu_begin = 0.0, cpu_end = 0.0, wall_begin = 0.0, wall_end = 0.0;
    double cpu1 = 0.0, cpu2 = 0.0, wall1 = 0.0, wall2 = 0.0;
};

inline void OpenMPAlgoBase::run() noexcept
{
    /* 初始化 */
    initialize();
    /* 并行计算 */
#pragma omp parallel num_threads(num_of_threads)
    {
        initialize_local();
#pragma omp for schedule(dynamic, 1) nowait
        for (size_t i = begin_indx; i < end_indx; ++i)
        {
            do_task(i);
            /* 合并数据 */
            if (should_upload_data_force())
            {
                upload_data_force();
            }
            else
            {
                if (should_upload_data())
                {
                    upload_data();
                }
            }
        }
        finish_local();
    }
    /* 结束 */
    finish();
}

inline void OpenMPAlgoBase::initialize() noexcept
{
    cpu_begin  = get_cputime();
    wall_begin = get_wtime();
}

inline void OpenMPAlgoBase::get_time_begin() noexcept
{
    cpu1  = get_cputime();
    wall1 = get_wtime();
}

inline void OpenMPAlgoBase::get_time_end() noexcept
{
    cpu2  = get_cputime();
    wall2 = get_wtime();
}

inline void OpenMPAlgoBase::finish() noexcept
{
    cpu_end  = get_cputime();
    wall_end = get_wtime();
}

}; // namespace poly_example04

}; // namespace Training2024