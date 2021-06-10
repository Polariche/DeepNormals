// referenced https://github.com/chrischoy/pytorch_knn_cuda/blob/master/src/knn_cuda_kernel.cu

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define SUBMATRIX_SIZE   32

namespace {


  template <typename scalar_t>
  __global__ void dist_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> x,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> y,
    const int nx,
    const int ny,
    const int c,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dist) {

    __shared__ scalar_t xs[SUBMATRIX_SIZE][SUBMATRIX_SIZE];
    __shared__ scalar_t ys[SUBMATRIX_SIZE][SUBMATRIX_SIZE];

    scalar_t temp;
    scalar_t sum = 0;

    // locate my dim & ind
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // for copying
    const int m = blockIdx.x * SUBMATRIX_SIZE + tx;
    const int n = blockIdx.y * SUBMATRIX_SIZE + tx;

    int m_within = m < nx;
    int n_within = n < ny;

    // for computation
    const int a = blockIdx.x * SUBMATRIX_SIZE + tx;
    const int b = blockIdx.y * SUBMATRIX_SIZE + ty;

    // is my location within the boundary, indicated by nx and ny?
    int a_within = a < nx;
    int b_within = b < ny;

    for (int i = 0; i < c; i += SUBMATRIX_SIZE) {
      // copy device memory to shared memory
      // tx points to the data index, ty points to the channel index
      if (i + ty < c) {
        xs[tx][ty] = m_within ? x[m][i + ty] : 0;
        ys[tx][ty] = n_within ? y[n][i + ty] : 0;

      } else {
        // channel is out-of-bounds
        xs[tx][ty] = 0;
        ys[tx][ty] = 0;
      }
      
      __syncthreads();

      // compute dist
      // tx points to x, ty points to y
      for (int j=0;j<SUBMATRIX_SIZE;j++) {
        temp = xs[tx][j] - ys[ty][j];
        sum += temp*temp;
      }

      __syncthreads();
    }

    if (a_within && b_within)
      dist[a][b] = a * ny + b;//sqrt(sum);

  }

    template <typename scalar_t>
  __global__ void sort_cuda_kernel(
    const int nx,
    const int ny,
    const int c,
    const int k,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dist_origin,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dist,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> ind) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < nx) {
      int j,w, ws,we;
      scalar_t cur_dist;

      cur_dist = dist_origin[i][0];

      dist[i][0] = dist_origin[i][0];
      ind[i][0] = 0;
      
      
      for(j=1;j<ny;j++) {
        cur_dist = dist_origin[i][j];
      
        ws = 0;
        we = j-1 < k-1? j-1 : k-1;
        
        while(we > ws) {  // binary search until the sub-array is indivisible
          w = (we + ws) / 2;
          if (dist[i][w] > cur_dist) {
            we = w - 1;
          } else {
            ws = w + 1;
          }
        }
        
        if (dist[i][ws] >= cur_dist) {
          // shift everything
          we = j < k-1? j : k-1;
          for (w=we;w>ws;w--) {
            dist[i][w] = dist[i][w-1];
            ind[i][w] = ind[i][w-1];
          }
          dist[i][ws] = cur_dist;
          ind[i][ws] = j;
        } else if (j < k) {
          dist[i][j] = cur_dist;
          ind[i][j] = j;
        }
       
      }
       
    }
    
  }

  // https://github.com/fanhqme/PointSetGeneration/blob/master/depthestimate/tf_nndistance_g.cu
  template <typename scalar_t>
  __global__ void dist_cuda_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_dist,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> x,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> y,
    const int nx,
    const int ny,
    const int c,
    const int k,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dist,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> ind,
    
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_x,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_y) {

    // one x's k-neighbors * c channels = 1 thread
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    int j, w;

    if (i < nx) {
      for (j = 0; j<k; j++) {
        scalar_t g = (1 / dist[i][j]) * d_dist[i][j];   // d(dist)^2 / d(dist) * dL / d(dist); chain rule
        int t = (int) ind[i][j];

        for (w=0; w<c; w++) {
          scalar_t d = x[i][w] - y[t][w];

          d_x[i][w] += d * g;
          //atomicAdd(&(d_x[i][w]), d * g);
          atomicAdd(&(d_y[t][w]), - d * g);
        }
      }
    }
    
  }

}

std::vector<torch::Tensor> knn_cuda_forward(
    torch::Tensor x, 
    torch::Tensor y, 
    int k) {

  const auto nx = x.size(0);
  const auto cx = x.size(1);

  const auto ny = y.size(0);
  const auto cy = y.size(1);

  assert (cx == cy);

  //auto options = torch::TensorOptions().dtype(x.type()).device(x.device(), 1);
  auto dist_origin = torch::zeros({nx, ny}, x.type());
  auto dist = torch::zeros({nx, k}, x.type());
  auto ind = torch::zeros({nx, k}, x.type());
  

  const int sm = SUBMATRIX_SIZE;
  const int sm2 = SUBMATRIX_SIZE*SUBMATRIX_SIZE;

  const dim3 d_blocks(nx/sm + nx%sm?1:0, ny/sm + ny%sm?1:0, 1);
  const dim3 s_blocks(nx/sm2 + nx%sm2?1:0, 1, 1);

  const dim3 d_threads(sm, sm, 1);
  const dim3 s_threads(sm2, 1, 1);

  // compute distance
  AT_DISPATCH_FLOATING_TYPES(x.type(), "dist_forward_cuda", ([&] {
    dist_cuda_forward_kernel<scalar_t><<<d_blocks, d_threads>>>(
        x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        y.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        nx,
        ny,
        cx,
        dist_origin.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
  }));



  // sort
  AT_DISPATCH_FLOATING_TYPES(dist_origin.type(), "sort_cuda", ([&] {
    sort_cuda_kernel<scalar_t><<<s_blocks, s_threads>>>(
        nx,
        ny,
        cx,
        k,
        dist_origin.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        dist.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        ind.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
  }));


  // sqrt


  return {dist, ind};
}


std::vector<torch::Tensor> knn_cuda_backward(
  torch::Tensor d_dist,
  torch::Tensor x, 
  torch::Tensor y,
  torch::Tensor dist,
  torch::Tensor ind) {

  const auto nx = x.size(0);
  const auto cx = x.size(1);

  const auto ny = y.size(0);
  const auto cy = y.size(1);

  const auto k = d_dist.size(1);

  assert (cx == cy);

  auto d_x = torch::zeros({nx, cx}, x.type());
  auto d_y = torch::zeros({ny, cx}, x.type());

  const int sm = SUBMATRIX_SIZE;
  const int sm2 = SUBMATRIX_SIZE*SUBMATRIX_SIZE;

  const dim3 s_blocks(nx/sm2 + nx%sm2?1:0, 1, 1);
  const int threads = sm2;

  AT_DISPATCH_FLOATING_TYPES(x.type(), "dist_backward_cuda", ([&] {
    dist_cuda_backward_kernel<scalar_t><<<s_blocks, threads>>>(
        d_dist.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        y.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        nx,
        ny,
        cx,
        k,
        dist.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        ind.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        
        d_x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        d_y.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
  }));

  return {d_x, d_y};
}

