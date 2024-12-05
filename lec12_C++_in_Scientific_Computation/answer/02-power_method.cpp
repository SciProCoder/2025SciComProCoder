#include "../include/basic.h"
#include "random_generator.h"

//----------------------------------------------------------------------
// abstract basic class for Matrix vector product
//----------------------------------------------------------------------

// 抽象基类：定义矩阵-向量乘法接口
template <typename Scalar>
class MVP
{
public:
    virtual ~MVP()                                                                                       = default;
    virtual void                apply(const std::vector<Scalar>& vec, std::vector<Scalar>& result) const = 0;
    virtual std::vector<Scalar> to_fullmat()                                                             = 0;
};

//----------------------------------------------------------------------
// Driver for power method
//----------------------------------------------------------------------

template <typename Scalar>
class PowerMethod
{
public:
    PowerMethod(MVP<Scalar>& mvp, size_t maxIterations, double tolerance) : mvp_(mvp), maxIterations_(maxIterations), tolerance_(tolerance) {}

    void run(const std::vector<Scalar>& initialVec)
    {
        std::vector<Scalar> vec = initialVec;

        auto norm_init = NORM2_VEC(vec.size(), vec.data(), 1);
        SCAL(vec.size(), 1.0 / norm_init, vec.data(), 1);

        std::vector<Scalar> nextVec(vec.size());
        Scalar              energy = 0, prevEnergy = 0;

        for (size_t iter = 0; iter < maxIterations_; ++iter)
        {
            mvp_.apply(vec, nextVec);

            prevEnergy = energy;
            energy     = 0;
            for (size_t i = 0; i < vec.size(); ++i)
            {
                energy += CONJ(nextVec[i]) * vec[i];
                // what will happen if we use
                // energy += std::conj(nextVec[i]) * vec[i]; ?
            }

            // 记录当前能量
            std::cout << "Iteration " << iter + 1 << ": Energy = " << energy << "\n";

            // 检查收敛性
            if (std::abs(energy - prevEnergy) < tolerance_)
            {
                std::cout << "Converged after " << iter + 1 << " iterations.\n";
                break;
            }

            vec.swap(nextVec);

            auto norm = NORM2_VEC(vec.size(), vec.data(), 1);
            SCAL(vec.size(), 1.0 / norm, vec.data(), 1);
            norm = sqrt(NORM2_VEC(vec.size(), vec.data(), 1));
        }
    }

private:
    MVP<Scalar>& mvp_;
    size_t       maxIterations_;
    double       tolerance_;
};

//----------------------------------------------------------------------
// Case 1 Dense matrix implementation
//----------------------------------------------------------------------

// 稠密矩阵实现
template <typename Scalar>
class DenseMVP : public MVP<Scalar>
{
public:
    // 构造函数：生成随机 Hermitian 矩阵
    explicit DenseMVP(size_t size) { _generate_matrix(size); }

    void apply(const std::vector<Scalar>& vec, std::vector<Scalar>& result) const override
    {
        for (size_t i = 0; i < matrix_.size(); ++i)
        {
            result[i] = 0;
            for (size_t j = 0; j < matrix_[i].size(); ++j)
            {
                result[i] += matrix_[i][j] * vec[j];
            }
        }
    }

    virtual std::vector<Scalar> to_fullmat()
    {
        std::vector<Scalar> fullmat;
        for (const auto& row : matrix_)
        {
            fullmat.insert(fullmat.end(), row.begin(), row.end());
        }
        return fullmat;
    }

private:
    std::vector<std::vector<Scalar>> matrix_;

    void _generate_matrix(size_t size) { matrix_ = generateRandomHermitianMatrix(size, Scalar{}); }
};

int main()
{
    static const int N = 8;

    // complex case

    using complex_t = std::complex<double>;

    DenseMVP<complex_t>    dense_mvp(N);
    std::vector<complex_t> InitVec(N, 0.0);
    InitVec[0] = 1.0;

    PowerMethod<complex_t> powerMethod(dense_mvp, 1000, 1e-6);
    powerMethod.run(InitVec);

    // real case

    using real_t = float;

    DenseMVP<real_t>    dense_mvp_real(N);
    std::vector<real_t> InitVecReal(N, 0.0);
    InitVecReal[0] = 1.0;

    PowerMethod<real_t> powerMethodReal(dense_mvp_real, 1000, 1e-6);
    powerMethodReal.run(InitVecReal);

    // check the result

    auto fullmat = dense_mvp.to_fullmat();

    // PRINT_MATRIX(fullmat.data(), N, N);

    std::vector<double>    eigval(N);
    std::vector<complex_t> eigvec(N * N);

    HermMatDiag(LAPACK_ROW_MAJOR, 'V', 'U', N, (lapack_complex_double*)fullmat.data(), N, eigval.data());

    printf("Eigenvalues:\n");
    for (const auto& val : eigval)
    {
        printf("%15.8f\n", val);
    }

    auto fullmat_real = dense_mvp_real.to_fullmat();

    // PRINT_MATRIX(fullmat_real.data(), N, N);

    std::vector<real_t> eigval_real(N);
    std::vector<real_t> eigvec_real(N * N);

    HermMatDiag(LAPACK_ROW_MAJOR, 'V', 'U', N, fullmat_real.data(), N, eigval_real.data());

    printf("Eigenvalues:\n");
    for (const auto& val : eigval_real)
    {
        printf("%15.8f\n", val);
    }

    return 0;
}