# gpu_shm_microbench

This benchmark mainly reflects the reading bandwidth of shared memory.

## Overview

The tables show the time(ms) cost of each case.

float(b32)

|                                         | a100    | v100    | mi50*   |
| --------------------------------------- | ------- | ------- | ------- |
| normal(tx)                              | 97.67   | 109.48  | 186.33  |
| multicast_2_way(tx/2)                   | 100.29  | 110.12  | 187.21  |
| multicast_4_way(tx/4)                   | 88.39   | 109.48  | 187.57  |
| multicast_8_way(tx/8)                   | 88.49   | 109.47  | 188.02  |
| multicast_16_way(tx/16)                 | 88.57   | 109.48  | 186.71  |
| boardcast(tx/32)                        | 88.58   | 109.47  | 187.81  |
| conflict_2_way( (tx%2) * 32 + tx/2 )    | 174.89  | 218.42  | 350.64  |
| conflict_4_way( (tx%4) * 32 + tx/4 )    | 349.27  | 454.38  | 699.11  |
| conflict_8_way( (tx%8) * 32 + tx/8 )    | 698.27  | 923.58  | 1394.66 |
| conflict_16_way( (tx%16) * 32 + tx/16 ) | 1395.27 | 1846.68 | 2784.43 |

float2(b64)

|                                         | a100    | v100    | mi50*   |
| --------------------------------------- | ------- | ------- | ------- |
| normal(tx)                              | 186.98  | 222.72  | 357.44  |
| multicast_2_way(tx/2)                   | 119.28  | 141.55  | 356.58  |
| multicast_4_way(tx/4)                   | 110.79  | 141.42  | 355.23  |
| multicast_8_way(tx/8)                   | 108.39  | 140.83  | 355.87  |
| multicast_16_way(tx/16)                 | 108.47  | 140.76  | 356.78  |
| boardcast(tx/32)                        | 108.47  | 141.52  | 354.98  |
| conflict_2_way( (tx%2) * 32 + tx/2 )    | 349.93  | 436.79  | 668.62  |
| conflict_4_way( (tx%4) * 32 + tx/4 )    | 698.38  | 883.00  | 1339.57 |
| conflict_8_way( (tx%8) * 32 + tx/8 )    | 1395.65 | 1836.83 | 2690.85 |
| conflict_16_way( (tx%16) * 32 + tx/16 ) | 2790.90 | 3693.49 | 5359.26 |

float4(b128)

|                                         | a100    | v100    | mi50*   |
| --------------------------------------- | ------- | ------- | ------- |
| normal(tx)                              | 375.11  | 453.53  | 715.56  |
| multicast_2_way(tx/2)                   | 214.08  | 263.57  | 715.16  |
| multicast_4_way(tx/4)                   | 204.77  | 263.54  | 716.34  |
| multicast_8_way(tx/8)                   | 194.95  | 263.80  | 715.64  |
| multicast_16_way(tx/16)                 | 191.59  | 263.69  | 715.70  |
| boardcast(tx/32)                        | 191.94  | 263.74  | 714.45  |
| conflict_2_way( (tx%2) * 32 + tx/2 )    | 699.18  | 873.59  | 1333.72 |
| conflict_4_way( (tx%4) * 32 + tx/4 )    | 1396.44 | 1817.17 | 2667.13 |
| conflict_8_way( (tx%8) * 32 + tx/8 )    | 2791.13 | 3694.27 | 2668.34 |
| conflict_16_way( (tx%16) * 32 + tx/16 ) | 2791.60 | 3694.27 | 2666.15 |

## Kernel

```cpp
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
```

## CUDA

```bash
nvcc test_cuda.cu -std=c++17 -O3
```

V100 output

```
warm up
finish warm up

float
normal(tx)              : time 109.481628(ms)
multicast_2_way(tx/2)           : time 110.128319(ms)
multicast_4_way(tx/4)           : time 109.484962(ms)
multicast_8_way(tx/8)           : time 109.478592(ms)
multicast_16_way(tx/16)         : time 109.485603(ms)
boardcast(tx/32)                : time 109.479713(ms)
conflict_2_way( (tx%2) * 32 + tx/2 )            : time 218.427490(ms)
conflict_4_way( (tx%4) * 32 + tx/4 )            : time 454.384857(ms)
conflict_8_way( (tx%8) * 32 + tx/8 )            : time 923.581299(ms)
conflict_16_way( (tx%16) * 32 + tx/16 )         : time 1846.681763(ms)

float2
normal(tx)              : time 222.728546(ms)
multicast_2_way(tx/2)           : time 141.557220(ms)
multicast_4_way(tx/4)           : time 141.428391(ms)
multicast_8_way(tx/8)           : time 140.837311(ms)
multicast_16_way(tx/16)         : time 140.760391(ms)
boardcast(tx/32)                : time 141.527161(ms)
conflict_2_way( (tx%2) * 32 + tx/2 )            : time 436.794525(ms)
conflict_4_way( (tx%4) * 32 + tx/4 )            : time 883.005554(ms)
conflict_8_way( (tx%8) * 32 + tx/8 )            : time 1836.837158(ms)
conflict_16_way( (tx%16) * 32 + tx/16 )         : time 3693.496094(ms)

float4
normal(tx)              : time 453.532318(ms)
multicast_2_way(tx/2)           : time 263.572845(ms)
multicast_4_way(tx/4)           : time 263.549133(ms)
multicast_8_way(tx/8)           : time 263.802734(ms)
multicast_16_way(tx/16)         : time 263.692383(ms)
boardcast(tx/32)                : time 263.745270(ms)
conflict_2_way( (tx%2) * 32 + tx/2 )            : time 873.591675(ms)
conflict_4_way( (tx%4) * 32 + tx/4 )            : time 1817.176270(ms)
conflict_8_way( (tx%8) * 32 + tx/8 )            : time 3694.271729(ms)
conflict_16_way( (tx%16) * 32 + tx/16 )         : time 3694.278320(ms)
```

A100 output

```
warm up
finish warm up

float
normal(tx)              : time 97.675133(ms)
multicast_2_way(tx/2)           : time 100.294403(ms)
multicast_4_way(tx/4)           : time 88.396927(ms)
multicast_8_way(tx/8)           : time 88.495743(ms)
multicast_16_way(tx/16)         : time 88.577667(ms)
boardcast(tx/32)                : time 88.583649(ms)
conflict_2_way( (tx%2) * 32 + tx/2 )            : time 174.897797(ms)
conflict_4_way( (tx%4) * 32 + tx/4 )            : time 349.275146(ms)
conflict_8_way( (tx%8) * 32 + tx/8 )            : time 698.276184(ms)
conflict_16_way( (tx%16) * 32 + tx/16 )         : time 1395.270386(ms)

float2
normal(tx)              : time 186.980057(ms)
multicast_2_way(tx/2)           : time 119.285027(ms)
multicast_4_way(tx/4)           : time 110.794403(ms)
multicast_8_way(tx/8)           : time 108.392670(ms)
multicast_16_way(tx/16)         : time 108.477699(ms)
boardcast(tx/32)                : time 108.473373(ms)
conflict_2_way( (tx%2) * 32 + tx/2 )            : time 349.934998(ms)
conflict_4_way( (tx%4) * 32 + tx/4 )            : time 698.380371(ms)
conflict_8_way( (tx%8) * 32 + tx/8 )            : time 1395.652466(ms)
conflict_16_way( (tx%16) * 32 + tx/16 )         : time 2790.907471(ms)

float4
normal(tx)              : time 375.114868(ms)
multicast_2_way(tx/2)           : time 214.085403(ms)
multicast_4_way(tx/4)           : time 204.778366(ms)
multicast_8_way(tx/8)           : time 194.950302(ms)
multicast_16_way(tx/16)         : time 191.595230(ms)
boardcast(tx/32)                : time 191.947174(ms)
conflict_2_way( (tx%2) * 32 + tx/2 )            : time 699.181763(ms)
conflict_4_way( (tx%4) * 32 + tx/4 )            : time 1396.447021(ms)
conflict_8_way( (tx%8) * 32 + tx/8 )            : time 2791.133545(ms)
conflict_16_way( (tx%16) * 32 + tx/16 )         : time 2791.607666(ms)
```

## ROCM

```bash
hipcc test_hip.cpp -std=c++17 -O3
```

MI50 plus output
```
warm up
finish warm up

float
normal(tx)              : time 186.338531(ms)
multicast_2_way(tx/2)           : time 187.210205(ms)
multicast_4_way(tx/4)           : time 187.570526(ms)
multicast_8_way(tx/8)           : time 188.026215(ms)
multicast_16_way(tx/16)         : time 186.719330(ms)
boardcast(tx/32)                : time 187.811646(ms)
conflict_2_way( (tx%2) * 32 + tx/2 )            : time 350.648102(ms)
conflict_4_way( (tx%4) * 32 + tx/4 )            : time 699.119324(ms)
conflict_8_way( (tx%8) * 32 + tx/8 )            : time 1394.666992(ms)
conflict_16_way( (tx%16) * 32 + tx/16 )         : time 2784.430420(ms)

float2
normal(tx)              : time 357.445068(ms)
multicast_2_way(tx/2)           : time 356.587769(ms)
multicast_4_way(tx/4)           : time 355.234497(ms)
multicast_8_way(tx/8)           : time 355.877991(ms)
multicast_16_way(tx/16)         : time 356.783905(ms)
boardcast(tx/32)                : time 354.982300(ms)
conflict_2_way( (tx%2) * 32 + tx/2 )            : time 668.626587(ms)
conflict_4_way( (tx%4) * 32 + tx/4 )            : time 1339.570923(ms)
conflict_8_way( (tx%8) * 32 + tx/8 )            : time 2690.853271(ms)
conflict_16_way( (tx%16) * 32 + tx/16 )         : time 5359.263184(ms)

float4
normal(tx)              : time 715.567261(ms)
multicast_2_way(tx/2)           : time 715.163635(ms)
multicast_4_way(tx/4)           : time 716.343140(ms)
multicast_8_way(tx/8)           : time 715.644043(ms)
multicast_16_way(tx/16)         : time 715.702942(ms)
boardcast(tx/32)                : time 714.454590(ms)
conflict_2_way( (tx%2) * 32 + tx/2 )            : time 1333.729370(ms)
conflict_4_way( (tx%4) * 32 + tx/4 )            : time 2667.139404(ms)
conflict_8_way( (tx%8) * 32 + tx/8 )            : time 2668.341309(ms)
conflict_16_way( (tx%16) * 32 + tx/16 )         : time 2666.159424(ms)
```

## ASM

### CUDA

sass be like

```asm
.L_x_0:
/*0240*/                   LDS R6, [R5.X4] ;
/*0250*/                   IADD3 R4, R4, 0x4, RZ ;
/*0260*/                   LDS R7, [R5.X4+0x800] ;
/*0270*/                   ISETP.NE.AND P0, PT, R4, 0x2710, PT ;
/*0280*/                   LDS R9, [R5.X4+0x1000] ;
/*0290*/                   LDS R11, [R5.X4+0x1800] ;
/*02a0*/                   LDS R13, [R5.X4+0x2000] ;
/*02b0*/                   LDS R15, [R5.X4+0x2800] ;
/*02c0*/                   LDS R17, [R5.X4+0x3000] ;
/*02d0*/                   LDS R19, [R5.X4+0x3800] ;
/*02e0*/                   FADD R6, RZ, R6 ;
/*02f0*/                   FADD R6, R6, R7 ;
/*0300*/                   FADD R6, R6, R9 ;
/*0310*/                   FADD R6, R6, R11 ;
/*0320*/                   FADD R6, R6, R13 ;
/*0330*/                   FADD R6, R6, R15 ;
/*0340*/                   FADD R6, R6, R17 ;
/*0350*/                   FADD R19, R6, R19 ;
/*0360*/                   STS [R0.X4], R19 ;
/*0370*/                   LDS R6, [R5.X4] ;
/*0380*/                   LDS R7, [R5.X4+0x800] ;
/*0390*/                   LDS R9, [R5.X4+0x1000] ;
/*03a0*/                   LDS R11, [R5.X4+0x1800] ;
/*03b0*/                   LDS R13, [R5.X4+0x2000] ;
/*03c0*/                   LDS R15, [R5.X4+0x2800] ;
/*03d0*/                   LDS R17, [R5.X4+0x3000] ;
/*03e0*/                   LDS R21, [R5.X4+0x3800] ;
/*03f0*/                   FADD R6, RZ, R6 ;
/*0400*/                   FADD R6, R6, R7 ;
/*0410*/                   FADD R6, R6, R9 ;
/*0420*/                   FADD R6, R6, R11 ;
/*0430*/                   FADD R6, R6, R13 ;
/*0440*/                   FADD R6, R6, R15 ;
/*0450*/                   FADD R6, R6, R17 ;
/*0460*/                   FADD R21, R6, R21 ;
/*0470*/                   STS [R0.X4], R21 ;
```

### ROCM

AMDGPU asm be like

```asm
.LBB0_1:                                ; %.preheader
                                        ; =>This Inner Loop Header: Depth=1
	ds_read_b32 v4, v0
	ds_read_b32 v5, v0 offset:2048
	ds_read_b32 v6, v0 offset:4096
	ds_read_b32 v7, v0 offset:6144
	ds_read_b32 v8, v0 offset:8192
	ds_read_b32 v9, v0 offset:10240
	ds_read_b32 v10, v0 offset:12288
	ds_read_b32 v11, v0 offset:14336
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v4, 0, v4
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v4, v4, v5
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v4, v4, v6
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v4, v4, v7
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v4, v4, v8
	s_add_i32 s1, s0, -4
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v4, v4, v9
	s_and_b32 s1, s1, 7
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v4, v4, v10
	v_lshl_or_b32 v13, s1, 11, v3
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v4, v4, v11
	ds_write_b32 v13, v4
```

