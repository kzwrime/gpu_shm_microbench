# gpu_shm_microbench

## CUDA

```bash
nvcc test_cuda.cu -std=c++17 -O3
```

## ROCM

```bash
hipcc test_hip.cpp -std=c++17 -O3
```
