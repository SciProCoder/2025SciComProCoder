#pragma once

#include "basic.h"

namespace Training2024 {
namespace concept_example {
template <typename T>
concept MatrixView = requires(T t) {
    typename T::scalar_type;
    requires std::floating_point<typename T::scalar_type>;
    typename T::iter_type;
    T::constant;
    T::scalar_size;
    { t.nrow() } -> std::convertible_to<size_t>;
    { t.ncol() } -> std::convertible_to<size_t>;
    { t.data() } -> std::convertible_to<typename T::iter_type>;
    { t.stride() } -> std::convertible_to<size_t>;
};

template <typename T>
concept CIVecView = requires(T t) {
    { t.nstates() } -> std::convertible_to<size_t>;
    { t.dimension() } -> std::convertible_to<size_t>;
} && MatrixView<T>;

template <std::floating_point T>
class MatrixView_1
{
public:
    // using iter_type = T;
    using scalar_type                   = T;
    static constexpr bool   constant    = false;
    static constexpr size_t scalar_size = sizeof(T);

    MatrixView_1(T *data, size_t nrow, size_t ncol, size_t stride = 1) : data_(data), nrow_(nrow), ncol_(ncol), stride_(stride) {}

    size_t nrow() const { return nrow_; }
    size_t ncol() const { return ncol_; }
    T     *data() const { return data_; }
    size_t stride() const { return stride_; }

protected:
    T     *data_   = nullptr;
    size_t nrow_   = 0;
    size_t ncol_   = 0;
    size_t stride_ = 0;
};

template <std::floating_point T>
class MatrixView_2
{
public:
    using iter_type                     = T;
    using scalar_type                   = T;
    static constexpr bool   constant    = false;
    static constexpr size_t scalar_size = sizeof(T);

    MatrixView_2(T *data, size_t nrow, size_t ncol, size_t stride = 1) : data_(data), nrow_(nrow), ncol_(ncol), stride_(stride) {}

    size_t nrow() const { return nrow_; }
    size_t ncol() const { return ncol_; }
    T     *data() const { return data_; }
    size_t stride() const { return stride_; }

protected:
    T     *data_   = nullptr;
    size_t nrow_   = 0;
    size_t ncol_   = 0;
    size_t stride_ = 0;
};

template <std::floating_point T>
class MatrixView_3
{
public:
    using iter_type                     = T *;
    using scalar_type                   = T;
    static constexpr bool   constant    = false;
    static constexpr size_t scalar_size = sizeof(T);

    MatrixView_3(T *data, size_t nrow, size_t ncol, size_t stride = 1) : data_(data), nrow_(nrow), ncol_(ncol), stride_(stride) {}

    size_t nrow() const { return nrow_; }
    size_t ncol() const { return ncol_; }
    T     *data() const { return data_; }
    size_t stride() const { return stride_; }

protected:
    T     *data_   = nullptr;
    size_t nrow_   = 0;
    size_t ncol_   = 0;
    size_t stride_ = 0;
};

template <std::floating_point T>
class CIVecView_1 : public MatrixView_3<T>
{
public:
    CIVecView_1(T *data, size_t nstates, size_t dimension, size_t stride = 1) : MatrixView_3<T>(data, nstates, dimension, stride) {}

    size_t nstates() const { return this->nrow(); }
    size_t dimension() const { return this->ncol(); }
};

template <MatrixView T1, MatrixView T2>
    requires std::same_as<typename T1::scalar_type, typename T2::scalar_type>
void Check_MatrixView(const T1 &t1, const T2 &t2)
{
    std::cout << "Check passed" << std::endl;
}

template <CIVecView T1, CIVecView T2>
    requires std::same_as<typename T1::scalar_type, typename T2::scalar_type>
void Check_CIVecView(const T1 &t1, const T2 &t2)
{
    std::cout << "Check passed" << std::endl;
}

}; // namespace concept_example

}; // namespace BDF_programmer_training