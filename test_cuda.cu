#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

// Size of array
#define N 128 * 80 * 16

#define WARM_UP_LOOP 100
#define KERNEL_LOOP 10
#define KERNEL_INNER_REPEAT 1000

const int thr_per_blk = 128;
const int blk_in_grid = ceil(double(N) / thr_per_blk);

enum share_read {
  normal,
  boardcast,
  multicast_2_way,
  multicast_4_way,
  multicast_8_way,
  multicast_16_way,
  conflict_2_way,
  conflict_4_way,
  conflict_8_way,
  conflict_16_way
};

#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__

HOST_DEVICE_INLINE float2 operator+(const float2 &u, const float2 &v) {
  return make_float2(u.x + v.x, u.y + v.y);
}

HOST_DEVICE_INLINE void operator+=(float2 &u, const float2 &v) {
  u.x += v.x;
  u.y += v.y;
}

HOST_DEVICE_INLINE float4 operator+(const float4 &u, const float4 &v) {
  return make_float4(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w);
}

HOST_DEVICE_INLINE void operator+=(float4 &u, const float4 &v) {
  u.x += v.x;
  u.y += v.y;
  u.z += v.z;
  u.w += v.w;
}

template <typename T> HOST_DEVICE_INLINE T MAKE_MTYPE(const float &v) {
  if constexpr (std::is_same_v<T, float>)
    return v;
  else if constexpr (std::is_same_v<T, float2>)
    return make_float2(v, v);
  else if constexpr (std::is_same_v<T, float4>)
    return make_float4(v, v, v, v);
  else
    static_assert("Unsupported type T in MAKE_MTYPE");
}

template <typename T> HOST_DEVICE_INLINE float GET_MTYPE(const T &v) {
  if constexpr (std::is_same_v<T, float>)
    return v;
  else if constexpr (std::is_same_v<T, float2>)
    return v.x * v.y;
  else if constexpr (std::is_same_v<T, float4>)
    return v.x * v.y * v.z * v.w;
  else
    static_assert("Unsupported type T in MAKE_MTYPE");
}

// Kernel
template <int choose, typename MTYPE> __global__ void add_vectors(float *a) {
  const int len = 1024 * sizeof(float4) / sizeof(MTYPE);
  __shared__ MTYPE shm[len];
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = threadIdx.x; i < len; i += blockDim.x)
    shm[i] = MAKE_MTYPE<MTYPE>(a[id]);

  MTYPE sum = MAKE_MTYPE<MTYPE>(0);
  for (int i = 0; i < KERNEL_INNER_REPEAT; i++) {
    for (int j = 0; j < 256; j++) {
      if constexpr (choose == normal) {
        sum += shm[threadIdx.x + j];
      } else if constexpr (choose == boardcast) {
        sum += shm[threadIdx.x / 32 + j];
      } else if constexpr (choose == multicast_2_way) {
        sum += shm[threadIdx.x / 2 + j];
      } else if constexpr (choose == multicast_4_way) {
        sum += shm[threadIdx.x / 4 + j];
      } else if constexpr (choose == multicast_8_way) {
        sum += shm[threadIdx.x / 8 + j];
      } else if constexpr (choose == multicast_16_way) {
        sum += shm[threadIdx.x / 16 + j];
      } else if constexpr (choose == conflict_2_way) {
        // 0->0, 1->32, 2->1, 3->33
        sum += shm[((threadIdx.x % 2) * 32) + threadIdx.x / 2 + j];
      } else if constexpr (choose == conflict_4_way) {
        // 0->0, 1->32, 2->64, 3->128; 4->1, 5->33, 6->65, 7->129;
        sum += shm[((threadIdx.x % 4) * 32) + threadIdx.x / 4 + j];
      } else if constexpr (choose == conflict_8_way) {
        sum += shm[((threadIdx.x % 8) * 32) + threadIdx.x / 8 + j];
      } else if constexpr (choose == conflict_16_way) {
        sum += shm[((threadIdx.x % 16) * 32) + threadIdx.x / 16 + j];
      }
    }
    shm[threadIdx.x + i % 256] = sum;
  }

  a[id] = GET_MTYPE(sum);
}

template <int choose, typename MTYPE>
void kernel_run(float *A, float *d_A, size_t bytes, const char *method) {
  cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
  float time_elapsed = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start); // 创建Event
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  for (int i = 0; i < KERNEL_LOOP; i++) {
    add_vectors<choose, MTYPE><<<blk_in_grid, thr_per_blk>>>(d_A);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start); // Waits for an event to complete.
  cudaEventSynchronize(stop); // Waits for an event to complete.Record之前的任务
  cudaEventElapsedTime(&time_elapsed, start, stop); // 计算时间差
  printf("%s: time %f(ms)\n", method, time_elapsed);
}

template <typename MTYPE> void run(float *A, float *d_A, size_t bytes) {

  kernel_run<normal, MTYPE>(A, d_A, bytes, "normal(tx)\t\t");
  kernel_run<multicast_2_way, MTYPE>(A, d_A, bytes, "multicast_2_way(tx/2)\t\t");
  kernel_run<multicast_4_way, MTYPE>(A, d_A, bytes, "multicast_4_way(tx/4)\t\t");
  kernel_run<multicast_8_way, MTYPE>(A, d_A, bytes, "multicast_8_way(tx/8)\t\t");
  kernel_run<multicast_16_way, MTYPE>(A, d_A, bytes, "multicast_16_way(tx/16)\t\t");
  kernel_run<boardcast, MTYPE>(A, d_A, bytes, "boardcast(tx/32)\t\t");
  kernel_run<conflict_2_way, MTYPE>(A, d_A, bytes, "conflict_2_way( (tx%2) * 32 + tx/2 )\t\t");
  kernel_run<conflict_4_way, MTYPE>(A, d_A, bytes, "conflict_4_way( (tx%4) * 32 + tx/4 )\t\t");
  kernel_run<conflict_8_way, MTYPE>(A, d_A, bytes, "conflict_8_way( (tx%8) * 32 + tx/8 )\t\t");
  kernel_run<conflict_16_way, MTYPE>(A, d_A, bytes, "conflict_16_way( (tx%16) * 32 + tx/16 )\t\t");

  cudaMemcpy(A, d_A, bytes, cudaMemcpyDeviceToHost);
}

// Main program
int main() {
  size_t bytes = N * sizeof(float);

  float *A = (float *)malloc(bytes);

  float *d_A;
  cudaMalloc(&d_A, bytes);

  for (int i = 0; i < N; i++) {
    A[i] = 1 + i / 1000.0;
  }

  cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);

  printf("warm up\n");
  // warm up
  for (int i = 0; i < WARM_UP_LOOP; i++) {
    add_vectors<normal, float><<<blk_in_grid, thr_per_blk>>>(d_A);
  }
  cudaDeviceSynchronize();
  printf("finish warm up\n");

  printf("\nfloat\n");
  run<float>(A, d_A, bytes);
  printf("\nfloat2\n");
  run<float2>(A, d_A, bytes);
  printf("\nfloat4\n");
  run<float4>(A, d_A, bytes);

  // Free CPU memory
  free(A);

  // Free GPU memory
  cudaFree(d_A);

  return 0;
}