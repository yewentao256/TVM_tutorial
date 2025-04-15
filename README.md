# TVM_tutorial

TVM tutorial for beginners.

See My Blog Series [Wentao.site](https://wentao.site/categories/tvm/) for more details.

## Conv 1d CPU

[conv1d_cpu.ipynb](conv1d_cpu.ipynb)

On an x86 CPU (“llvm” target) With `M = 4096, N = 128`. Key versions:

| **Implementation**      | **Time (ms)** | **Speedup vs. Naive** |
|-------------------------|---------------|------------------------|
| **Naive**               | ~24.35        | 1.0×                  |
| **v0** (Smaller k)      | ~23.05        | 1.06×                 |
| **v1** (Parallel)       | ~22.92        | 1.06×                 |
| **v2** (+Vectorize)     | ~16.04        | 1.52×                 |
| **v3** (+Unroll)        | ~14.60        | 1.67×                 |
| **v4** (Refactor)       | ~0.57         | 43.03×                |
| **v5** (Combined)       | ~0.41         | 59.39×                |
| **AutoTVM** (50 trials) | ~0.70         | 34.79×                |
| **NumPy**               | ~0.21         | ~116× (vs. Naive)     |

## Conv 1d GPU

[conv1d_gpu.ipynb](conv1d_gpu.ipynb)

On a Tesla T4 GPU with `M = 16384` and `N = 32`. Key versions:

| **Implementation**    | **Time (ms)** | **Speedup vs. Naive** | **Speedup vs. Previous** |
|-----------------------|---------------|------------------------|--------------------------|
| **Naive**             | 18.286        | 1.0×                   | -                        |
| **v1** (Refactor)     | 0.107         | 170.9×                 | 170.9×                   |
| **v2** (Threads)      | 0.0251        | 728.5×                 | 4.3×                     |
| **v3** (2D Threads)   | 0.0158        | 1157.3×                | 1.6×                     |
| **v4** (Memory Hier.) | 0.0147        | 1244.0×                | 1.1×                     |
| **v5** (+ Unroll)     | 0.0124        | 1474.7×                | 1.2×                     |
| **AutoTVM**       | 0.0405 ms        | 451.5×                |      -                     |
| **NumPy** (CPU)       | 0.2369        | 77.2×                  | -                        |
| **PyTorch** (GPU)     | 0.1491        | 122.6×                 | -                        |

## Conv 2d GPU

[conv2d_dw_gpu.ipynb](conv2d_dw_gpu.ipynb)

For a 2D depthwise convolution with parameters `B=3, C=4, H=16, W=32, K=7` on a GPU:

| **Implementation**       | **Time (ms)** | **Speedup vs. Naive** | **Speedup vs. Previous** |
|--------------------------|---------------|------------------------|--------------------------|
| **Naive**                | 3.2904        | 1.0×                   | -                        |
| **v1** (2D Blocks)       | 0.7687        | 2.3×                   | 2.3×                     |
| **v2** (Block Fusion)    | 0.0762        | 43.2×                  | 10.1×                    |
| **v3** (2D Threads)      | 0.0101        | 325.8×                 | 7.5×                     |
| **v4** (Outer Fusion)    | 0.0080        | 411.3×                 | 1.26×                    |
| **AutoTVM**              | 0.0035        | 940.1×                 | 2.29×                    |
| **PyTorch**         | 0.0696        | 47.3×                  | -                        |

## GEMM GPU

[gemm.ipynb](gemm.ipynb)

For a matrix multiplication with `M=1024`, `K=2048`, `N=512` on a GPU:

| **Implementation**       | **Time (ms)** | **Speedup vs. Naive** | **Speedup vs. Previous** |
|--------------------------|---------------|------------------------|--------------------------|
| **Naive**                | 84.52         | 1.0×                   | -                        |
| **v1** (1D Threads)      | 36.98         | 2.3×                   | 2.3×                     |
| **v2** (2D Threads)      | 35.50         | 2.4×                   | 1.04×                    |
| **v3** (Shared Memory)   | 8.11          | 10.4×                  | 4.4×                     |
| **v4** (Combined)   | 4.56          | 18.5×                  | 1.8×                     |
| **AutoTVM**              | 42.56         | 2.0×                   | -                        |
| **NumPy** (CPU)          | 74.95         | 1.1×                   | -                        |
| **PyTorch CPU**          | 18.74         | 4.5×                   | -                        |
| **PyTorch CUDA**         | 0.70          | 120.7×                 | -                        |
