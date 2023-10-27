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

