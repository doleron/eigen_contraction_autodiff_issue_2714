// see https://eigen.tuxfamily.org/dox/TopicCUDA.html
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU

#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/AutoDiff>
#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>

typedef typename Eigen::AutoDiffScalar<Eigen::VectorXf> AutoDiff_T;

Eigen::Tensor<AutoDiff_T, 2> run_gpu_contraction(const Eigen::Tensor<AutoDiff_T, 2> &A, const Eigen::Tensor<AutoDiff_T, 2> &B);