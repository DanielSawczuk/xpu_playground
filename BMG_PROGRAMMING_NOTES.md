# Programming Intel BMG GPUs with SYCL, eSIMD, XMX, and Level Zero

These notes summarize the practical information collected while implementing
and measuring peak-memory and peak-BF16-compute kernels on an Intel Arc Pro B70
(BMG/Xe2). They describe behavior observed with the oneAPI 2026.0 compiler and
Level Zero driver `1.15.38308+4`; compiler and driver behavior can change.

## Tested device and reference rates

- Device: Intel Arc Pro B70 Graphics
- Compute units reported by SYCL: 256
- Supplied peak memory bandwidth: 608.0 GB/s
- Supplied peak BF16 XMX throughput: 183.0 TFLOP/s
- Level Zero vendor ID for Intel devices: `0x8086`

Measured results from the implementations in this directory were approximately:

| Host path | Memory bandwidth | BF16 XMX throughput |
|---|---:|---:|
| SYCL | 596.6 GB/s | 183.45 TFLOP/s |
| Level Zero | 597.4 GB/s | 183.45 TFLOP/s |

Both paths validate output checksums. Results depend on clocks, power limits,
temperature, allocation size, and the number of warmup and timed launches.

## Useful source material in this workspace

Reproduce the reference-repository layout with:

```sh
./checkout_source_material.sh
```

The default destination is the parent of `xpu_playground`. Pass a workspace
root as the first argument to use another location. The script checks out the
tested revisions listed below.

- eSIMD specification and extension documentation:
  `../llvm/sycl/doc/extensions/supported/sycl_ext_intel_esimd` at
  `intel/llvm@0ef90649443bdc833902a5fadbf4daf9bfe992d8`
- XeTLA, a template library built on eSIMD: `../xetla` at
  `intel/xetla@7a1acbde4ff608141500e142324923257605862a`
- Level Zero specification: `../level-zero-spec` at
  `oneapi-src/level-zero-spec@60c28d2051071fdd484a72306c43fa0519a515b0`
- Memory example: `peak_mem_2d_kernel.hpp` and `peak_mem_2d_kernel.cpp`
- XMX example: `peak_compute_dpas_kernel.hpp` and
  `peak_compute_dpas_kernel.cpp`
- Pure Level Zero host example: `level_zero/peak_l0.cpp`

The pinned XeTLA revision predates current SYCL headers and compiler APIs.
Apply `patches/xetla-oneapi-2026.patch` before building it; instructions and
the verified build command are in `patches/README.md`.

## Compiler targets

### Dumping eSIMD assembly and checking GRF mode

The IGC shader-dump controls also capture VC output for eSIMD kernels. For a
JIT build, disable the runtime cache so compilation actually occurs:

```sh
icpx -O3 -std=c++17 -fsycl -fsycl-device-code-split=per_kernel \
  gemm.cpp -o /tmp/gemm-jit
IGC_ShaderDumpEnable=1 \
IGC_DumpToCustomDir=/tmp/gemm-dumps \
IGC_ShaderDumpPidDisable=1 \
SYCL_CACHE_DISABLE=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
  /tmp/gemm-jit --m 2560 --n 2560 --k 32 --iterations 1 --warmups 1
```

For AOT, put the same `IGC_*` variables on the compiler invocation. Relevant
outputs are `VC_*.asm`, `VC_*.visaasm`, `VC_*.zeinfo`, and
`VC_*_options.txt`. Large-GRF mode is confirmed by all three of:

```text
options:       -doubleGRF -vc-codegen
zeinfo:        grf_count: 256
assembly:      .thread_config numGRF=256
```

With oneAPI 2026, the `grf_size<256>` kernel property was honored by JIT but
did not reach the `spir64_gen` AOT finalizer. The GEMM Makefile rule therefore
passes `-options '-doubleGRF'` explicitly. A scratch-location declaration is
part of the standard ABI and does not by itself indicate spilling; look for
actual scratch instructions or the finalizer's `Spill memory used` message.

For an AOT SYCL executable targeting the BMG family:

```sh
icpx -O3 -std=c++17 -fsycl \
  -fsycl-targets=spir64_gen \
  -Xsycl-target-backend=spir64_gen "-device bmg" \
  -fsycl-device-code-split=per_kernel \
  host.cpp kernel.cpp -o program
```

`bmg-g31` can be used when an exact G31 target is desired. The `bmg` target is
the more general family target used by this project's Makefile.

For portable SPIR-V, use the complete DPC++ offload pipeline rather than only
translating the free-function definition:

```sh
dpcpp -O3 -std=c++17 -fsycl \
  -fsycl-targets=spir64 \
  -fsycl-device-code-split=per_kernel \
  level_zero/peak_mem_spv_entry.cpp peak_mem_2d_kernel.cpp \
  -o temporary_bundle
```

The compiler bundle contains multiple SPIR-V images. This project uses
`clang-offload-extract` and selects the image containing the dedicated Level
Zero dispatch entry. Run `make level-zero` to perform this automatically.

The `dpcpp` command is deprecated in oneAPI 2026 and prints a warning; its
supported replacement is `icpx -fsycl`. It remains in the SPIR-V rules here
because the requested build path explicitly uses DPC++.

## Free-function eSIMD kernels

A separately compiled free-function kernel is declared and defined with these
properties:

```cpp
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::nd_range_kernel<1>))
SYCL_EXTERNAL SYCL_ESIMD_KERNEL
void kernel_name(const float *input, float *output, unsigned scalar);
```

Important details:

- `SYCL_EXTERNAL` permits the host and kernel definition to live in separate
  translation units.
- `SYCL_ESIMD_KERNEL` selects the explicit-SIMD compilation path.
- `nd_range_kernel<1>` describes the free function as a one-dimensional
  ND-range kernel.
- The declaration visible to the host must have exactly the same properties
  and signature as the definition.
- The SYCL host launches it with `nd_launch` and
  `kernel_function<kernel_name>`.
- Pointer and scalar argument order becomes part of the kernel ABI and must be
  preserved by a direct Level Zero launcher.

Compiling only the annotated free-function definition with
`-fsycl-device-only -fsycl-device-obj=spirv` produced SPIR-V, but on the tested
toolchain it did not produce a kernel that the Level Zero VC path exposed for
dispatch. The full DPC++ post-link pass performs eSIMD lowering and emits the
dispatch kernel. The small `level_zero/*_spv_entry.cpp` files instantiate that
dispatch wrapper while continuing to call the same free-function
implementation.

## Block-2D memory operations

The bandwidth kernel uses `esimd::load_2d`, not a collection of scalar or
ordinary vector loads. Its tested geometry is:

- Surface width: 1024 `float` elements, or a 4096-byte pitch
- Block: 16 columns by 8 rows of `float`
- Bytes per block-2D load: `16 * 8 * 4 = 512` bytes
- Loads per eSIMD work-item: 4
- Read traffic per work-item: 2048 bytes
- Local size: 16 eSIMD work-items

Four independent loads at successive eight-row offsets provide enough
outstanding traffic while keeping the useful traffic accounting simple. The
tested cache properties are L1 streaming and L2 cached:

```cpp
constexpr esimd::properties properties{
    esimd::cache_hint_L1<esimd::cache_hint::streaming>,
    esimd::cache_hint_L2<esimd::cache_hint::cached>};
```

The block-2D API takes width, height, and pitch in the instruction's encoded
form. In these kernels the surface width in bytes, surface height, and pitch
are passed as `value - 1`, as required by the API overload being used.

The work-item coordinate mapping is:

```text
x = (global_id % tiles_per_row) * block_width
y = (global_id / tiles_per_row) * rows_per_work_item
```

With 32 rows consumed per work-item position, the allocation must be a
multiple of `4096 bytes/row * 32 rows = 128 KiB`. The kernel has no tail path;
the host must guarantee that every 2D block is in bounds. The scalar kernel
argument is the surface height in rows, not the total number of elements.

### Avoid memory-compression artifacts

Do not initialize a bandwidth surface with one repeated value. Intel GPU
lossless memory compression can represent uniform data compactly and make an
apparent bandwidth result exceed the physical DRAM rate. The benchmark uses a
deterministic integer hash to generate irregular finite floats before timing.
Initialization and copies are excluded from the measured interval.

Count only the intended traffic. This benchmark reports the 2 KiB of reads per
work-item and does not add the small checksum store to the bandwidth numerator.

## BF16 DPAS/XMX programming

The compute kernel uses `sycl::ext::oneapi::bfloat16` inputs and FP32
accumulators. The tested Xe2/BMG operation is:

```text
C[M,N] += A[M,K] * B[K,N]
M = 8, N = 16, K = 16
```

One DPAS instruction therefore represents:

```text
2 * M * N * K = 4096 floating-point operations
```

The key implementation details are:

- A is loaded as an 8-by-16 BF16 tile.
- B is stored as a 16-by-16 row-major tile and loaded with the block-2D
  transform needed for the vertical/VNNI DPAS operand layout:
  `load_2d<bf16, 16, 16, 1, false, true>`.
- Both operand tiles are issued through `prefetch_2d` before the block-2D
  loads.
- The tested operand surface has a 64-byte pitch. This satisfies the block-2D
  surface pitch requirement for these tiles.
- Eight independent FP32 accumulator chains cover DPAS latency and sustain the
  XMX issue rate.
- Different initial accumulator values prevent unwanted common-subexpression
  folding.
- `#pragma unroll 1` on the runtime rounds loop prevents code-size explosion
  while retaining the intended repeated DPAS sequence.

With eight DPAS operations per round, each work-item performs 32,768 FLOP per
round. The default launch uses 4096 work-items and 8192 rounds:

```text
4096 * 8192 * 8 * 4096 = 1.0995e12 FLOP per launch
```

Filling A and B with BF16 `0.5` gives a predictable DPAS result. The raw BF16
bit representation of `0.5` is `0x3f00`, which is convenient for a Level Zero
memory fill.

## Pure Level Zero host path

The Level Zero host executable does not link the SYCL runtime. It is built as
ordinary C++ and links only the loader:

```sh
c++ -O3 -std=c++17 level_zero/peak_l0.cpp -lze_loader \
  -o level_zero/peak_l0
```

The essential initialization sequence is:

1. Call `zeInit(ZE_INIT_FLAG_GPU_ONLY)`.
2. Enumerate drivers and devices, selecting an Intel GPU.
3. Create a context.
4. Find a queue group with
   `ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE`.
5. Create an asynchronous immediate command list.
6. Load the extracted SPIR-V with `ZE_MODULE_FORMAT_IL_SPIRV`.
7. Create the kernel, set group size 16, and set arguments in declaration
   order.

eSIMD SPIR-V must be passed through the VC backend when Level Zero creates the
module:

```cpp
module_desc.pBuildFlags = "-vc-codegen -disable-finalizer-msg";
```

Without `-vc-codegen`, the tested driver attempted its normal subgroup path
and module compilation failed with an unsupported required-subgroup-size
error.

### Kernel names

DPC++ generates mangled dispatch entry names. Query them with
`zeModuleGetKernelNames` when the driver supports it. The tested driver could
return an empty list for some SPIR-V IL modules, so the runner also carries the
known generated name and accepts `--kernel NAME` as an override. Do not assume
that the C++ free-function symbol is necessarily the dispatchable kernel name.

### Kernel arguments and group geometry

- For a pointer argument, call `zeKernelSetArgumentValue` with
  `sizeof(void *)` and the address of the device-pointer variable.
- For a scalar, pass its exact ABI size and address.
- Set `zeKernelSetGroupSize(kernel, 16, 1, 1)` before launch.
- Convert the logical global size to `ze_group_count_t` by dividing by 16.
- Round or validate work-item counts so the global size is divisible by the
  local size. The memory geometry is exact and should be validated rather than
  silently padded.

The benchmark uses `zeMemAllocDevice` for GPU data and `zeMemAllocHost` for a
staging buffer. Memory initialization is copied in chunks so a 1 GiB benchmark
does not require a second 1 GiB host allocation.

## Accurate timing

Use kernel timestamp events rather than host wall-clock time:

```cpp
pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE |
                  ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
```

After launch and `zeEventHostSynchronize`, call
`zeEventQueryKernelTimestamp` and use the global start/end timestamps.

Request `ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2` when querying device
properties. With this structure version, `timerResolution` is the timer
frequency in cycles per second, so elapsed seconds are:

```text
masked_cycle_delta / timerResolution
```

Mask the subtraction with `kernelTimestampValidBits` to handle counter wrap.
Older device-property structure versions define `timerResolution`
differently, so mixing the two interpretations produces incorrect timings.

For SYCL, enable queue profiling and use event `command_start` and
`command_end`. In both APIs:

- Perform allocation and initialization before timing.
- Use several warmups to raise clocks and populate instruction caches.
- Report at least the median; the best result is also useful for identifying
  clock-management outliers.
- Use enough work and iterations. Very short runs fluctuate substantially and
  are not reliable peak measurements.

## Validation and performance checklist

- Verify eSIMD and block-2D support before using the SYCL path.
- Use device allocations for the timed working set.
- Validate all block-2D pitch, width, height, alignment, and bounds
  requirements for the selected element type and tile shape.
- Ensure host scalar arguments use the units expected by the kernel.
- Use irregular memory contents when measuring physical DRAM bandwidth.
- Keep a checksum or another observable dependency so the compiler cannot
  delete loads or DPAS work.
- Use multiple independent memory operations or accumulator chains to cover
  hardware latency.
- Keep initialization, submission overhead, and result copies outside the
  kernel timestamp.
- Check both the first and last work-item results to catch ABI and geometry
  mistakes.
- Compare against the correct operation-count convention: one fused multiply
  and add is two FLOP.
- Re-run after driver or compiler updates; generated entry names and JIT
  behavior are implementation details rather than stable application ABI.

## Commands for this project

Build and run the SYCL path:

```sh
make
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./peak_mem_2d
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./peak_compute_dpas
```

Build and run the pure Level Zero path:

```sh
make level-zero
./level_zero/peak_l0 mem
./level_zero/peak_l0 compute
```

Shorter validation runs:

```sh
./peak_mem_2d --size-mib 512 --iterations 5
./peak_compute_dpas --iterations 5
./level_zero/peak_l0 mem --size-mib 512 --iterations 5
./level_zero/peak_l0 compute --iterations 5
```
