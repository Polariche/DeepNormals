// referenced https://github.com/chrischoy/pytorch_knn_cuda/blob/master/src/knn_cuda_kernel.cu

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define SUBMATRIX_SIZE   32

namespace {


  template <typename scalar_t>
  __global__ void compute_face_features_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> vertices,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> faces,
    const int ny,
    //const int c,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> face_offset,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> N,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> A,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> B,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dots,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> X_N) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    const int j0 = (int) faces[i][0];
    const int j1 = (int) faces[i][1];
    const int j2 = (int) faces[i][2];

    const scalar_t X0 = vertices[j0][0];
    const scalar_t X1 = vertices[j0][1];
    const scalar_t X2 = vertices[j0][2];

    const scalar_t A0 = X0 - vertices[j1][0];
    const scalar_t A1 = X1 - vertices[j1][1];
    const scalar_t A2 = X2 - vertices[j1][2];

    const scalar_t B0 = X0 - vertices[j2][0];
    const scalar_t B1 = X1 - vertices[j2][1];
    const scalar_t B2 = X2 - vertices[j2][2];

    const scalar_t N0 = A1 * B2 - A2 * B1;
    const scalar_t N1 = A2 * B0 - A0 * B2;
    const scalar_t N2 = A1 * B2 - A2 * B1;

    const scalar_t dots0 = A0*A0 + A1*A1 + A2*A2;
    const scalar_t dots1 = A0*B0 + A1*B1 + A2*B2;
    const scalar_t dots2 = B0*B0 + B1*B1 + B2*B2;

    const scalar_t XN = X0*N0 + X1*N1 + X2*N2;


    face_offset[i][0] = X0;
    face_offset[i][1] = X1;
    face_offset[i][2] = X2;  

    N[i][0] = N0;
    N[i][1] = N1;
    N[i][2] = N2;  

    A[i][0] = A0;
    A[i][1] = A1;
    A[i][2] = A2;  

    B[i][0] = B0;
    B[i][1] = B1;
    B[i][2] = B2;  
    
    dots[i][0] = dots0;
    dots[i][1] = dots1;
    dots[i][2] = dots2; 

    X_N[i] = XN;
  }

  template <typename scalar_t>
  __global__ void compute_face_features_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> ray,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> offset,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> face_offset,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> N,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> X_N,
    const int nx,
    const int ny,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> t,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> intersection) {

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    const int j_ = blockIdx.y * blockDim.y + threadIdx.x;   // for copying

    __shared__ scalar_t r[SUBMATRIX_SIZE][3];
    __shared__ scalar_t o[SUBMATRIX_SIZE][3];

    __shared__ scalar_t f[SUBMATRIX_SIZE][3];
    __shared__ scalar_t n[SUBMATRIX_SIZE][3];
    __shared__ scalar_t x_n[SUBMATRIX_SIZE];

    scalar_t t;
    scalar_t hit[3];

    // copying memory
    if (ty < 3) {
        if (i < nx) {
            r[tx][ty] = ray[i][ty];
            o[tx][ty] = offset[i][ty];
        } else {
            r[tx][ty] = 0;
            o[tx][ty] = 0;
        }

        if (j_ < ny) {
            f[tx][ty] = face_offset[j_][ty];
            n[tx][ty] = N[j_][ty];

        } else {
            f[tx][ty] = 0;
            n[tx][ty] = 0;
        }
    }

    if (ty == 0) {
        if (j_ < ny)
            x_n[tx] = X_N[j_];
        else
            x_n[tx] = 0;
    }

    __syncthreads();

    if (i < nx && j < ny) {
        // we find t s.t (t*r + o - f) * n = 0
        scalar_t div = (r[tx][0] * n[ty][0] + r[tx][1] * n[ty][1] + r[tx][2] * n[ty][2]);

        if (div < 1e-6 && div > 1e-6) {
            t = 0;
            hit[0] = 0;
            hit[1] = 0;
            hit[2] = 0;
        }
        t = (x_n[ty] 
            + o[tx][0] * n[ty][0] + o[tx][1] * n[ty][1] + o[tx][2] * n[ty][2])
            / (r[tx][0] * n[ty][0] + r[tx][1] * n[ty][1] + r[tx][2] * n[ty][2]);

        hit[0] = t * r[tx][0] + o[tx][0] - f[ty][0];
        hit[1] = t * r[tx][1] + o[tx][1] - f[ty][1];
        hit[2] = t * r[tx][2] + o[tx][2] - f[ty][2];

    } else {
        t = 0;
        hit[0] = 0;
        hit[1] = 0;
        hit[2] = 0;
    }
        

  }
}

std::vector<torch::Tensor> visibility_test(
    torch::Tensor ray, 
    torch::Tensor offset,
    torch::Tensor vertices, 
    torch::Tensor faces) {

  assert (ray.size(0) == offset.size(0));
  assert (ray.size(1) == offset.size(1));
  assert (offset.size(1) == vertices.size(1));
  assert (vertices.size(2) == 3);     // 3D only
  assert (faces.size(2) == 3);       // only consider trig faces

  const auto nx = ray.size(0);
  const auto ny = faces.size(0);
  //const auto c = ray.size(1);

  const int sm = SUBMATRIX_SIZE;
  const int sm2 = SUBMATRIX_SIZE*SUBMATRIX_SIZE;

  const dim3 d_blocks(nx/sm + (nx%sm?1:0), ny/sm + (ny%sm?1:0), 1);
  const dim3 s_blocks(ny/sm2 + (ny%sm2?1:0), 1, 1);

  const dim3 d_threads(sm, sm, 1);
  const dim3 s_threads(sm2, 1, 1);


  auto face_offset = torch::zeros({ny, c}, vertices.type());
  auto N = torch::zeros({ny, c}, vertices.type());
  auto A = torch::zeros({ny, c}, vertices.type());
  auto B = torch::zeros({ny, c}, vertices.type());
  auto dots = torch::zeros({ny, 3}, vertices.type());  // (A*A, A*B, B*B)
  auto X_N = torch::zeros({ny}, vertices.type());

  auto t = torch::zeros({nx, ny}, vertices.type());

  // compute face features
  AT_DISPATCH_FLOATING_TYPES(vertices.type(), "compute_face_features_kernel", ([&] {
    compute_face_features_kernel<scalar_t><<<s_blocks, s_threads>>>(
        vertices.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        faces.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        ny,
        //c,
        face_offset.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        N.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        A.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        B.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        dots.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        X_N.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>()
        );
  }));

  AT_DISPATCH_FLOATING_TYPES(vertices.type(), "compute_ray_distance_kernel", ([&] {
    compute_ray_distance_kernel<scalar_t><<<d_blocks, d_threads>>>(
        ray.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        offset.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        face_offset.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        N.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        X_N.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        nx,
        ny,
        );
  }));

  return {dist, ind};
}



