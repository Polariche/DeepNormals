#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> knn_cuda_forward(
    torch::Tensor x, 
    torch::Tensor y, 
    int k);

std::vector<torch::Tensor> knn_cuda_backward(
    torch::Tensor d_dist,
	torch::Tensor x, 
	torch::Tensor y, 
	torch::Tensor dist,
	torch::Tensor ind);

// C++ interface


// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_x(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)




std::vector<torch::Tensor> knn_forward(
    torch::Tensor x, 
    torch::Tensor y, 
    int k) {

  CHECK_x(x);
  CHECK_x(y);

  return knn_cuda_forward(x,y,k);
}

std::vector<torch::Tensor> knn_backward(
    torch::Tensor d_dist,
	torch::Tensor x, 
	torch::Tensor y,
	torch::Tensor dist,
	torch::Tensor ind) {

  //CHECK_x(d_dist);
  CHECK_x(x);
  CHECK_x(y);
  CHECK_x(dist);
  CHECK_x(ind);

  return knn_cuda_backward(d_dist, x, y, dist, ind);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &knn_forward, "KNN forward (CUDA)");
  m.def("backward", &knn_backward, "KNN backward (CUDA)");
}