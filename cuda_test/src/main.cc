#include "functions.hpp"

int main()
{

    const int S = 64;

    Eigen::Tensor<AutoDiff_T, 2> X(S/2, S);
    Eigen::Tensor<AutoDiff_T, 2> W(S, S/2);

    const int size = X.size() + W.size();

    for (size_t i = 0; i < X.dimension(0); ++i)
    {
        for (size_t j = 0; j < X.dimension(1); ++j)
        {
            X(i, j).value() = 1.f;
            X(i, j).derivatives() = Eigen::VectorXf::Unit(size, j + i * X.dimension(1));
        }
    }

    for (size_t i = 0; i < W.dimension(0); ++i)
    {
        for (size_t j = 0; j < W.dimension(1); ++j)
        {
            W(i, j).value() = 1.f;
            W(i, j).derivatives() = Eigen::VectorXf::Unit(size, X.size() + j + i * W.dimension(1));
        }
    }

    run_gpu_contraction(X, W);

    return 0;
}