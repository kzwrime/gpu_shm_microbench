# gpu_shm_microbench

## CUDA

```bash
nvcc test_cuda.cu -std=c++17 -O3
```

A100 output

```
warm up
finish warm up
no_conflict: time 663.738647(ms)
boardcast time 618.054443(ms)
multicast time 618.685181(ms)
conflict_2_way time 1160.160400(ms)
conflict_4_way time 2249.523438(ms)
conflict_8_way time 4430.911133(ms)
conflict_16_way time 8788.954102(ms)
```

## ROCM

```bash
hipcc test_hip.cpp -std=c++17 -O3
```

MI50 plus output
```
warm up
finish warm up
no_conflict: time 1065.581421(ms)
boardcast time 1065.825073(ms)
multicast time 1065.851685(ms)
conflict_2_way time 1901.468628(ms)
conflict_4_way time 3236.115967(ms)
conflict_8_way time 6349.378906(ms)
conflict_16_way time 14200.084961(ms)
```


