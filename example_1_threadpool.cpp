#define EIGEN_MALLOC_ALREADY_ALIGNED 1

#define EIGEN_USE_THREADS
#include <Eigen/ThreadPool>

#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/AutoDiff>

typedef typename Eigen::AutoDiffScalar<Eigen::VectorXf> AutoDiff_T;
typedef typename Eigen::Tensor<AutoDiff_T, 2>::DimensionPair DimPair;

int main(int, char **)
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

    const int threads = 4;
    Eigen::ThreadPool tp(threads);
    Eigen::ThreadPoolDevice device(&tp, threads);

    Eigen::array<DimPair, 1> dims;
    dims[0] = DimPair(1, 0);
    Eigen::Tensor<AutoDiff_T, 2> Z(X.dimension(0), W.dimension(1));
    Z.device(device) = X.contract(W, dims);

    return 0;
}