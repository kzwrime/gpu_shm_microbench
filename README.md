# gpu_shm_microbench

Default items_per_thread=8, which means 8 read + 8 add + 1 write, mainly reflects the reading performance of shared memory.

## Overview

float(b32)

|                | a100     | v100     | mi50*    |
| -------------- | -------- | -------- | -------- |
| no_conflict    | 658.3846 | 800.3199 | 1335.369 |
| boardcast      | 620.4108 | 802.95   | 1329.768 |
| multicast      | 620.5339 | 802.8201 | 1329.622 |
| conflict_2_way | 1159.872 | 1513.676 | 2218.167 |
| conflict_4_way | 2249.229 | 2946.2   | 4279.984 |
| conflict_8_way | 4429.086 | 5798.598 | 8421.638 |

float2(b64)

|                | a100     | v100     | mi50*    |
| -------------- | -------- | -------- | -------- |
| no_conflict    | 1290.588 | 1733.207 | 2474.466 |
| boardcast      | 688.8486 | 942.9678 | 2475.849 |
| multicast      | 688.9868 | 948.5419 | 2475.404 |
| conflict_2_way | 2318.283 | 3234.857 | 4441.515 |
| conflict_4_way | 4495.824 | 6347.18  | 8558.571 |
| conflict_8_way | 8858.317 | 12710.63 | 16831.07 |

float4(b128)

|                | a100     | v100     | mi50*    |
| -------------- | -------- | -------- | -------- |
| no_conflict    | 2575.202 | 3265.734 | 4612.925 |
| boardcast      | 1370.866 | 1713.209 | 4616.237 |
| multicast      | 1370.973 | 1855.482 | 4614.917 |
| conflict_2_way | 4640.534 | 6222.208 | 8868.114 |
| conflict_4_way | 8996.727 | 11784.03 | 17139.82 |
| conflict_8_way | 17716.83 | 22991.49 | 17139.83 |

## CUDA

```bash
nvcc test_cuda.cu -std=c++17 -O3
```

V100 output

```
# float
warm up
finish warm up
no_conflict: time 800.319946(ms)
boardcast time 802.950012(ms)
multicast time 802.820129(ms)
conflict_2_way time 1513.675659(ms)
conflict_4_way time 2946.200439(ms)
conflict_8_way time 5798.598145(ms)

#float2
warm up
finish warm up
no_conflict: time 1733.207153(ms)
boardcast time 942.967834(ms)
multicast time 948.541870(ms)
conflict_2_way time 3234.857178(ms)
conflict_4_way time 6347.179688(ms)
conflict_8_way time 12710.625000(ms)

#float4
warm up
finish warm up
no_conflict: time 3265.734131(ms)
boardcast time 1713.209106(ms)
multicast time 1855.482422(ms)
conflict_2_way time 6222.208496(ms)
conflict_4_way time 11784.026367(ms)
conflict_8_way time 22991.486328(ms)
```

A100 output

```
#float
warm up
finish warm up
no_conflict: time 658.384644(ms)
boardcast time 620.410828(ms)
multicast time 620.533875(ms)
conflict_2_way time 1159.872314(ms)
conflict_4_way time 2249.229492(ms)
conflict_8_way time 4429.085938(ms)

#float2
warm up
finish warm up
no_conflict: time 1290.588135(ms)
boardcast time 688.848572(ms)
multicast time 688.986755(ms)
conflict_2_way time 2318.283447(ms)
conflict_4_way time 4495.824219(ms)
conflict_8_way time 8858.317383(ms)

#float4
warm up
finish warm up
no_conflict: time 2575.201904(ms)
boardcast time 1370.865723(ms)
multicast time 1370.972778(ms)
conflict_2_way time 4640.534180(ms)
conflict_4_way time 8996.726562(ms)
conflict_8_way time 17716.828125(ms
```

## ROCM

```bash
hipcc test_hip.cpp -std=c++17 -O3
```

MI50 plus output
```
#float
warm up
finish warm up
no_conflict: time 1335.369141(ms)
boardcast time 1329.768433(ms)
multicast time 1329.622192(ms)
conflict_2_way time 2218.166504(ms)
conflict_4_way time 4279.983887(ms)
conflict_8_way time 8421.637695(ms)

#float2
warm up
finish warm up
no_conflict: time 2474.466309(ms)
boardcast time 2475.849121(ms)
multicast time 2475.404053(ms)
conflict_2_way time 4441.515137(ms)
conflict_4_way time 8558.571289(ms)
conflict_8_way time 16831.074219(ms)

#float4
warm up
finish warm up
no_conflict: time 4612.925293(ms)
boardcast time 4616.236816(ms)
multicast time 4614.916504(ms)
conflict_2_way time 8868.114258(ms)
conflict_4_way time 17139.820312(ms)
conflict_8_way time 17139.826172(ms)
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

