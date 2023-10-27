# gpu_shm_microbench

default items_per_thread=8

## CUDA

```bash
nvcc test_cuda.cu -std=c++17 -O3
```

V100 output

```
warm up
finish warm up
no_conflict: time 850.027893(ms)
boardcast time 857.593018(ms)
multicast time 856.612244(ms)
conflict_2_way time 1629.663208(ms)
conflict_4_way time 3162.770752(ms)
conflict_8_way time 6319.376953(ms)
conflict_16_way time 12723.411133(ms)
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
no_conflict: time 1338.479614(ms)
boardcast time 1338.416870(ms)
multicast time 1338.822510(ms)
conflict_2_way time 2209.756592(ms)
conflict_4_way time 4279.312500(ms)
conflict_8_way time 8420.298828(ms)
conflict_16_way time 16683.246094(ms)
```


