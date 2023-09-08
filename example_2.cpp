#include <unsupported/Eigen/CXX11/Tensor>

typedef typename Eigen::Tensor<float, 2>::DimensionPair DimPair;

int main(int, char **)
{

    Eigen::Tensor<float, 2> X(2, 2);
    Eigen::Tensor<float, 2> W(2, 2);

    const int size = X.size() + W.size();

    X(0, 0) = 1.f;
    X(1, 0) = 1.f;
    X(0, 1) = 1.f;
    X(1, 1) = 1.f;

    W(0, 0) = 1.f;
    W(1, 0) = 1.f;
    W(0, 1) = 1.f;
    W(1, 1) = 1.f;

    Eigen::array<DimPair, 1> dims;
    dims[0] = DimPair(1, 0);
    Eigen::Tensor<float, 2> Z = X.contract(W, dims);

    return 0;
}