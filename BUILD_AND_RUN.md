# Flux Build & Run Guide (2x H800 PCIe)

This doc records how to build and run Flux on the current machine (2x NVIDIA H800 PCIe, SM count 114).

## 1. Environment

```text
OS:        Ubuntu 22.04
GPU:       2 x NVIDIA H800 PCIe
CUDA:      12.8 (/usr/local/cuda-12.8)
Driver:    580.82.07
Python:    3.12.3 (miniconda)
PyTorch:   2.8.0+cu128
```

## 2. Install Dependencies

```bash
# CUDA paths
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8

# Python build deps
pip install ninja packaging

# NVSHMEM (required for MoE kernels)
pip install nvidia-nvshmem-cu12==3.3.9

# Create symlink so the linker can find libnvshmem_host.so
NVSHMEM_HOME=$(python3 -c "import nvidia.nvshmem, pathlib; print(pathlib.Path(nvidia.nvshmem.__path__[0]))")
ln -s libnvshmem_host.so.3 ${NVSHMEM_HOME}/lib/libnvshmem_host.so

# System deps required by unit tests / flux build
apt-get update
apt-get install -y libnuma-dev libgflags-dev libgoogle-glog-dev
```

## 3. Clone & Init Submodules

```bash
git submodule update --init --recursive
```

This pulls `3rdparty/cutlass` and `3rdparty/nccl`.

## 4. Patch for SM Count 114

The H800s in this machine report 114 SMs, but Flux only knows `78/92/108/132`. Without the patch, runtime crashes with:

```text
Unsupported SM count for SMCoreEnum: 114
```

Patch `src/cuda/op_registry.cu` so that 114 is treated the same as 132 (H800):

```cpp
  switch (sm_count) {
    case 92: sm_core = SMCoreEnum::L20; break;
    case 108: sm_core = SMCoreEnum::A100; break;
    case 78: sm_core = SMCoreEnum::H20; break;
    case 114:
    case 132: sm_core = SMCoreEnum::H800; break;
    default: FLUX_CHECK(false) << "Unsupported SM count for SMCoreEnum: " << sm_count; break;
  }
```

## 5. Build Flux

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export NVSHMEM_HOME=$(python3 -c "import nvidia.nvshmem, pathlib; print(pathlib.Path(nvidia.nvshmem.__path__[0]))")

./build.sh --arch 90 --nvshmem
```

The build compiles:
- NCCL (3rdparty)
- `libflux_cuda.so`
- `libflux_cuda_ths_op.so`
- unit tests

Note: `setup.py develop --user` inside `build.sh` may fail in conda with `File exists: '/root/miniconda3/bin/python3.12'`. In that case, finish Python installation manually:

```bash
export FLUX_SHM_USE_NVSHMEM=1
pip install -e . --no-build-isolation
```

## 6. Verify Import

```bash
export FLUX_SHM_USE_NVSHMEM=1
python3 -c "import flux; print(flux.__file__)"
```

## 7. Run Examples / Tests

All multi-GPU commands below use the project-provided `launch.sh`, which auto-detects the 2 GPUs and invokes `torchrun`.

### Single-GPU GEMM-only

```bash
python3 test/python/gemm_only/test_gemm_only.py 4096 12288 6144 \
  --input_dtype float16 --weight_dtype float16
```

### 2-GPU All-Gather + GEMM

```bash
./launch.sh test/python/ag_gemm/test_ag_kernel.py 4096 49152 12288 \
  --dtype=float16 --iters=10
```

### 2-GPU MoE Layer0 (default EP=1, TP=2)

```bash
cd examples
../launch.sh moe_layer0.py --iters=3
```

This runs successfully. The output shows `all close!` and `✅ flux check passed`. A `bitwise match` warning is normal for bf16/fp16 workloads due to numerical differences between the PyTorch reference and the fused Flux kernel.

## 8. Known Limitations on This Machine

### GEMM + Reduce-Scatter on PCIe

```bash
./launch.sh test/python/gemm_rs/test_gemm_rs.py 4096 12288 49152 \
  --dtype=float16 --iters=10
```

Fails with:

```text
No registered hparams found ... comm_kind=IntraNodePcie
```

Cause: the two H800s are connected via PCIe, not NVLink. Flux's tuning configs only register `IntraNodeNvLink` variants for reduce-scatter on H800. There is no simple flag to switch the comm kind; fixing it requires either:

1. Adding `IntraNodePcie` entries to `src/gemm_rs/tuning_config/config_gemm_rs_sm90_H800_*.cu`, or
2. Running on a machine with NVLink-connected GPUs.

### MoE examples with hard-coded EP=4

`examples/moe.py` and `examples/moe_flux_only.py` hard-code `init_ep_group(ep_size=4)`. On a 2-GPU machine this assertion fails:

```text
DIST_ENV.WORLD_SIZE % ep_size != 0
```

To run them on 2 GPUs, edit the script to use `init_ep_group(ep_size=2)` or `init_ep_group(ep_size=1)` and adjust the `tp_env` / `MoeArguments` accordingly.

## 9. Quick Reference: Environment Variables

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export NVSHMEM_HOME=$(python3 -c "import nvidia.nvshmem, pathlib; print(pathlib.Path(nvidia.nvshmem.__path__[0]))")
export FLUX_SHM_USE_NVSHMEM=1
```

## 10. Clean Build

```bash
./build.sh --clean-all
rm -rf build
```

Then rebuild from step 5.
