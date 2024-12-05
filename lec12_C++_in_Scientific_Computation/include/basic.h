#pragma once

/* C lib */

#include <stdint.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <cstdlib>
#include <stddef.h>
#include <unistd.h>
#include <ctype.h>
#include <sys/mman.h> /* mmap munmap */
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>
#include <execinfo.h>

/* CXX Lib */

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <set>
#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <complex>
#include <chrono>
#include <map>
#include <type_traits>
#include <memory>
#include <random>
#include <concepts>

/* math configuration */

#define complex_double std::complex<double>
#define complex_float std::complex<float>

#ifdef _MKL_
#include <mkl.h>
#else

#include <cblas.h>
#include <lapacke.h>

#define MKL_INT CBLAS_INDEX
#define MKL_Complex16 std::complex<double>
#define MKL_Complex8 std::complex<float>
#endif

///// register std::complex<double> as float_point

namespace std {
template <>
struct __is_floating_point_helper<std::complex<double>> : public std::true_type
{
};
template <>
struct __is_floating_point_helper<std::complex<float>> : public std::true_type
{
};
}; // namespace std

///// some macro

#define GETTER(type, name) \
    type get_##name() const noexcept { return name; }

#define SETTER(type, name, classname)          \
    classname &set_##name(type input) noexcept \
    {                                          \
        name = input;                          \
        return *this;                          \
    }

#define DEFINE_VIRTUAL_FUNCTION(funcname) \
    virtual void funcname() noexcept {}

#define DECLARE_VIRTUAL_FUNCTION(funcname) virtual void funcname() noexcept;

#define DECLARE_PURE_VIRTUAL_FUNCTION(funcname) virtual void funcname() = 0;

#define OMP_NUM_OF_THREADS std::atoi(getenv("OMP_NUM_THREADS"))
#define OMP_THREAD_LABEL omp_get_thread_num()