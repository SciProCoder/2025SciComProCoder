#pragma once

#include <random>
#include "01-linear_algebra_interface.h"
using namespace Training2024::LinearAlgebraInterface;

std::vector<std::vector<float>> generateRandomHermitianMatrix(size_t size, float)
{
    std::vector<std::vector<float>>       matrix(size, std::vector<float>(size, 0.0f));
    std::mt19937                          rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = i; j < size; ++j)
        {
            if (i == j)
            {
                matrix[i][j] = dist(rng);
            }
            else
            {
                float value  = dist(rng);
                matrix[i][j] = value;
                matrix[j][i] = value;
            }
        }
    }
    return matrix;
}

std::vector<std::vector<double>> generateRandomHermitianMatrix(size_t size, double)
{
    std::vector<std::vector<double>>       matrix(size, std::vector<double>(size, 0.0));
    std::mt19937                           rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = i; j < size; ++j)
        {
            if (i == j)
            {
                matrix[i][j] = dist(rng);
            }
            else
            {
                double value = dist(rng);
                matrix[i][j] = value;
                matrix[j][i] = value;
            }
        }
    }
    return matrix;
}

std::vector<std::vector<std::complex<float>>> generateRandomHermitianMatrix(size_t size, std::complex<float>)
{
    std::vector<std::vector<std::complex<float>>> matrix(size, std::vector<std::complex<float>>(size));
    std::mt19937                                  rng(std::random_device{}());
    std::uniform_real_distribution<float>         realDist(-1.0f, 1.0f);
    std::uniform_real_distribution<float>         imagDist(-1.0f, 1.0f);

    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = i; j < size; ++j)
        {
            if (i == j)
            {
                matrix[i][j] = {realDist(rng), 0.0f};
            }
            else
            {
                std::complex<float> value = {realDist(rng), imagDist(rng)};
                matrix[i][j]              = value;
                matrix[j][i]              = std::conj(value);
            }
        }
    }
    return matrix;
}

std::vector<std::vector<std::complex<double>>> generateRandomHermitianMatrix(size_t size, std::complex<double>)
{
    std::vector<std::vector<std::complex<double>>> matrix(size, std::vector<std::complex<double>>(size));
    std::mt19937                                   rng(std::random_device{}());
    std::uniform_real_distribution<double>         realDist(-1.0, 1.0);
    std::uniform_real_distribution<double>         imagDist(-1.0, 1.0);

    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = i; j < size; ++j)
        {
            if (i == j)
            {
                matrix[i][j] = {realDist(rng), 0.0};
            }
            else
            {
                std::complex<double> value = {realDist(rng), imagDist(rng)};
                matrix[i][j]               = value;
                matrix[j][i]               = std::conj(value);
            }
        }
    }
    return matrix;
}
