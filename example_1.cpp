#include <iostream>

#define EIGEN_MALLOC_ALREADY_ALIGNED 1

#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/AutoDiff>


typedef typename Eigen::AutoDiffScalar<Eigen::VectorXf> AutoDiff_T;
typedef typename Eigen::Tensor<AutoDiff_T, 2>::DimensionPair DimPair;

int main(int, char **)
{

    Eigen::Tensor<AutoDiff_T, 2> X(2, 2);
    Eigen::Tensor<AutoDiff_T, 2> W(2, 2);

    const int size = X.size() + W.size();

    X(0, 0).value() = 1.f;
    X(0, 0).derivatives() = Eigen::VectorXf::Unit(size, 0);
    X(1, 0).value() = 1.f;
    X(1, 0).derivatives() = Eigen::VectorXf::Unit(size, 1);
    X(0, 1).value() = 1.f;
    X(0, 1).derivatives() = Eigen::VectorXf::Unit(size, 2);
    X(1, 1).value() = 1.f;
    X(1, 1).derivatives() = Eigen::VectorXf::Unit(size, 3);

    W(0, 0).value() = 1.f;
    W(0, 0).derivatives() = Eigen::VectorXf::Unit(size, 4);
    W(1, 0).value() = 1.f;
    W(1, 0).derivatives() = Eigen::VectorXf::Unit(size, 5);
    W(0, 1).value() = 1.f;
    W(0, 1).derivatives() = Eigen::VectorXf::Unit(size, 6);
    W(1, 1).value() = 1.f;
    W(1, 1).derivatives() = Eigen::VectorXf::Unit(size, 7);

    Eigen::array<DimPair, 1> dims;
    dims[0] = DimPair(1, 0);
    Eigen::Tensor<AutoDiff_T, 2> Z = X.contract(W, dims);

    std::cout << std::flush;

    return 0;
}