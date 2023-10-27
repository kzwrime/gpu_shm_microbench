#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

// Size of array
#define N 256 * 80 * 16

enum share_read {
  no_conflict,
  boardcast,
  multicast,
  conflict_2_way,
  conflict_4_way,
  conflict_8_way,
  conflict_16_way
};

// Kernel
template <int choose> __global__ void add_vectors(float *a) {
  const int ITEMS = 8;
  __shared__ float shm[ITEMS][512];
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = 0; i < ITEMS; i++)
    shm[i][threadIdx.x] = a[id] * i * threadIdx.x;

  for (int i = 0; i < 10000; i++) {
    float sum = 0;
    for (int j = 0; j < ITEMS; j++) {
      if constexpr (choose == no_conflict) {
        sum += shm[j][(threadIdx.x + 1) % 256];
      } else if constexpr (choose == boardcast) {
        sum += shm[j][threadIdx.x / 32];
      } else if constexpr (choose == multicast) {
        sum += shm[j][threadIdx.x / 4];
      } else if constexpr (choose == conflict_2_way) {
        // 0->0, 1->32, 2->1, 3->33
        sum += shm[j][((threadIdx.x % 2) * 32) + threadIdx.x / 2];
      } else if constexpr (choose == conflict_4_way) {
        // 0->0, 1->32, 2->64, 3->128;
        // 4->1, 5->33, 6->65, 7->129;
        sum += shm[j][((threadIdx.x % 4) * 32) + threadIdx.x / 4];
      } else if constexpr (choose == conflict_8_way) {
        sum += shm[j][((threadIdx.x % 8) * 32) + threadIdx.x / 8];
      } else if constexpr (choose == conflict_16_way) {
        sum += shm[j][((threadIdx.x % 16) * 32) + threadIdx.x / 16];
      }
    }
    shm[0][threadIdx.x] = sum;
  }

  a[id] = shm[7][threadIdx.x];
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

  int thr_per_blk = 256;
  int blk_in_grid = ceil(double(N) / thr_per_blk);

  printf("warm up\n");
  // warm up
  for (int i = 0; i < 1000; i++) {
    add_vectors<no_conflict><<<blk_in_grid, thr_per_blk>>>(d_A);
  }
  cudaDeviceSynchronize();
  printf("finish warm up\n");

  cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
  float time_elapsed = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start); // 创建Event
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  for (int i = 0; i < 100; i++) {
    add_vectors<no_conflict><<<blk_in_grid, thr_per_blk>>>(d_A);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start); // Waits for an event to complete.
  cudaEventSynchronize(stop); // Waits for an event to complete.Record之前的任务
  cudaEventElapsedTime(&time_elapsed, start, stop); // 计算时间差
  printf("no_conflict: time %f(ms)\n", time_elapsed);
  cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);

  cudaEventRecord(start, 0);
  for (int i = 0; i < 100; i++) {
    add_vectors<boardcast><<<blk_in_grid, thr_per_blk>>>(d_A);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start); // Waits for an event to complete.
  cudaEventSynchronize(stop); // Waits for an event to complete.Record之前的任务
  cudaEventElapsedTime(&time_elapsed, start, stop); // 计算时间差
  printf("boardcast time %f(ms)\n", time_elapsed);

  cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(start, 0);
  for (int i = 0; i < 100; i++) {
    add_vectors<multicast><<<blk_in_grid, thr_per_blk>>>(d_A);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start); // Waits for an event to complete.
  cudaEventSynchronize(stop); // Waits for an event to complete.Record之前的任务
  cudaEventElapsedTime(&time_elapsed, start, stop); // 计算时间差
  printf("multicast time %f(ms)\n", time_elapsed);

  cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(start, 0);
  for (int i = 0; i < 100; i++) {
    add_vectors<conflict_2_way><<<blk_in_grid, thr_per_blk>>>(d_A);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start); // Waits for an event to complete.
  cudaEventSynchronize(stop); // Waits for an event to complete.Record之前的任务
  cudaEventElapsedTime(&time_elapsed, start, stop); // 计算时间差
  printf("conflict_2_way time %f(ms)\n", time_elapsed);

  cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(start, 0);
  for (int i = 0; i < 100; i++) {
    add_vectors<conflict_4_way><<<blk_in_grid, thr_per_blk>>>(d_A);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start); // Waits for an event to complete.
  cudaEventSynchronize(stop); // Waits for an event to complete.Record之前的任务
  cudaEventElapsedTime(&time_elapsed, start, stop); // 计算时间差
  printf("conflict_4_way time %f(ms)\n", time_elapsed);

  cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(start, 0);
  for (int i = 0; i < 100; i++) {
    add_vectors<conflict_8_way><<<blk_in_grid, thr_per_blk>>>(d_A);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start); // Waits for an event to complete.
  cudaEventSynchronize(stop); // Waits for an event to complete.Record之前的任务
  cudaEventElapsedTime(&time_elapsed, start, stop); // 计算时间差
  printf("conflict_8_way time %f(ms)\n", time_elapsed);

  cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(start, 0);
  for (int i = 0; i < 100; i++) {
    add_vectors<conflict_16_way><<<blk_in_grid, thr_per_blk>>>(d_A);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start); // Waits for an event to complete.
  cudaEventSynchronize(stop); // Waits for an event to complete.Record之前的任务
  cudaEventElapsedTime(&time_elapsed, start, stop); // 计算时间差
  printf("conflict_16_way time %f(ms)\n", time_elapsed);

  cudaMemcpy(A, d_A, bytes, cudaMemcpyDeviceToHost);

  cudaMemcpy(A, d_A, bytes, cudaMemcpyDeviceToHost);
  // Free CPU memory
  free(A);

  // Free GPU memory
  cudaFree(d_A);

  return 0;
}
