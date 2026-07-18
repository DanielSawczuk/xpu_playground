# BMG eSIMD peak-throughput benchmarks

This directory contains two BMG peak-throughput benchmarks, GEMM, and GEMV:

- `peak_mem_2d.cpp`: memory benchmark host and profiling code.
- `peak_mem_2d_kernel.cpp`: separately compiled eSIMD free-function kernel
  using four independent block-2D loads.
- `peak_compute_dpas.cpp`: BF16 benchmark host and profiling code.
- `peak_compute_dpas_kernel.cpp`: separately compiled eSIMD free-function
  kernel using XMX/DPAS, block-2D prefetch, and block-2D operand loads.
- `gemm.cpp`: a standalone eSIMD/XMX row-major GEMM with FP32 accumulation, FP16 or BF16
  inputs/output, edge predication, and automatic padding for arbitrary sizes.
- `gemv.cpp`: a standalone row-major matrix-vector multiply using eSIMD
  block-2D reads, FP32 accumulation, and FP16 or BF16 inputs/output.

See [`BMG_PROGRAMMING_NOTES.md`](BMG_PROGRAMMING_NOTES.md) for the collected
implementation notes, compiler and Level Zero details, performance methodology,
and pitfalls found while developing these benchmarks.

Run `./checkout_source_material.sh` to check out the tested eSIMD, XeTLA, and
Level Zero specification repositories beside this repository.

After checkout, apply
[`patches/xetla-oneapi-2026.patch`](patches/xetla-oneapi-2026.patch) to compile
the archived XeTLA revision with oneAPI 2026. See
[`patches/README.md`](patches/README.md) for application and build commands.

`peak_mem_2d.cpp` measures sustained device-memory read bandwidth with native
eSIMD `load_2d` operations. Each eSIMD work-item issues four independent
16-by-8 `float` loads (2 KiB total) and writes one four-byte checksum. Reported
bandwidth counts read bytes only and compares them with the supplied B70 peak
of 608.0 GB/s.

Build with the oneAPI 2026 compiler:

```sh
source /opt/intel/oneapi/setvars.sh
make
```

The default AOT device, `bmg`, is compatible with the BMG family. An exact
device can be selected when desired, for example:

```sh
make clean
make DEVICE=bmg-g31
```

Run on the Level Zero GPU backend:

```sh
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./peak_sycl mem
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./peak_sycl compute
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./peak_sycl gemv --size 16384 --type bf16
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./peak_sycl gemm --size 4096 --type bf16
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./peak_mem_2d
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./peak_compute_dpas
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./gemm --size 4096 --type bf16
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./gemv --size 16384 --type bf16
```

## oneDNN interoperability

The same oneDNN matmul primitive can be submitted from either a SYCL host or
a pure Level Zero host. Both examples wrap the application's existing device,
context, queue/command list, and USM allocations; oneDNN does not create a
second GPU runtime. The runtime-independent primitive is kept in
`onednn_matmul.hpp`, leaving only the interoperability calls in each host.

Build against oneDNN 3.14 or newer (the direct Level Zero interoperability
header was added after oneDNN 3.11):

```sh
make onednn ONEDNN_ROOT=/path/to/onednn/install
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./onednn_sycl 4096 100 fp16
./level_zero/onednn_l0 4096 100 fp16
```

The optional third argument selects `f32` (the default), `bf16`, or `fp16`.
Both hosts use the runtime-specific interoperability execution API so event
dependencies for primitives with multiple internal kernels remain valid.

The SYCL target can also be built alone with the oneAPI 2026 oneDNN 3.11
installation: `make onednn_sycl`.

The GEMM defaults to a 4096-square BF16 multiply. It accepts independent
dimensions, FP16 input/output, and configurable timing counts:

```sh
./gemm --m 3072 --n 4096 --k 2048 --type fp16 --iterations 20 --warmups 5
```

The GEMM has no library dependency beyond SYCL/eSIMD. Its large workgroup
kernel uses 256-by-256 workgroup tiles, cooperative block-2D prefetches for L1
operand reuse, large-GRF mode, snake swizzling, and sixteen independent DPAS
chains in each 32-by-64 work-item tile; it does not stage operands through SLM.
The GEMM build rule also passes `-doubleGRF` directly to the AOT backend;
oneAPI 2026 does not propagate the kernel GRF property to the AOT finalizer.

GEMV computes a row-major `M x K` matrix times a `K`-element column vector.
Each eSIMD work-item streams one matrix row with 32-element block-2D reads,
keeps the vector cached, and accumulates in FP32. Irregular device-generated
inputs avoid lossless-memory-compression artifacts. For a large matrix its
arithmetic intensity is approximately one FLOP per matrix byte, so GFLOP/s and
effective matrix-read GB/s are nearly identical:

```sh
./gemv --m 16384 --k 16384 --type fp16 --iterations 50 --warmups 5
```

The memory defaults are a 1024 MiB surface, three warmups, and 100 timed
launches. The free-function kernel is declared in `peak_mem_2d_kernel.hpp`,
defined in `peak_mem_2d_kernel.cpp`, and launched with `nd_launch`.
They can be changed without recompiling:

```sh
./peak_mem_2d --size-mib 1024 --iterations 100
```

The input surface is initialized with irregular finite values before timing;
this prevents lossless memory compression from inflating the result above the
physical DRAM rate. Timing comes from SYCL event profiling (`command_start` to
`command_end`), so allocation, initialization, submission, and checksum-copy
time are excluded. The program checks both eSIMD and block-2D device support
before launching the kernel.

The compute benchmark uses eight independent FP32 accumulator chains. Each
round issues eight BF16 DPAS operations of shape 8-by-16-by-16, or 32,768 FLOP
per eSIMD work-item. The free-function kernel is declared in
`peak_compute_dpas_kernel.hpp`, defined in `peak_compute_dpas_kernel.cpp`, and
launched from the host with `nd_launch`. Its defaults are 4096 work-items, 8192
rounds, and 1000 timed launches; they can be changed as follows:

```sh
./peak_compute_dpas --work-items 4096 --rounds 8192 --iterations 1000
```

It reports throughput relative to the supplied B70 peak of 183.0 TFLOP/s.

## Level Zero host path

All four kernels can also be compiled to standalone device modules and
launched by a host executable that uses only the Level Zero API (no SYCL host
runtime):

```sh
make level-zero
./level_zero/peak_l0 mem
./level_zero/peak_l0 compute
./level_zero/peak_l0 gemv --size 16384 --type bf16
./level_zero/peak_l0 gemm --size 4096 --type bf16
```

The unified runners have matching `mem`, `compute`, `gemv`, and `gemm`
subcommands: `peak_sycl` uses the SYCL host API, while
`level_zero/peak_l0` uses only Level Zero. The original four SYCL executables
remain as compatibility entry points and accept the same options as their
subcommands. The SYCL dispatcher keeps the kernels in separately compiled AOT
images so GEMM can use `-doubleGRF` without forcing that performance-sensitive
setting onto GEMV and the peak kernels.

`make level-zero` runs the complete DPC++ device pipeline, including eSIMD
post-link lowering, and extracts dispatchable images from the temporary
compiler bundles. The small files `peak_mem_spv_entry.cpp` and
`peak_compute_spv_entry.cpp` instantiate Level Zero dispatch entries that call
the same separately maintained free-function kernels.

The peaks use SPIR-V modules. GEMV emits native FP16 and BF16 modules, while
GEMM additionally emits native small- and large-tile variants with the
required `-doubleGRF` finalizer option. The runner selects the same tile at the
2560-by-2560 threshold as the SYCL host. The Level Zero runner discovers the
compiler-generated entry name, passes
`-vc-codegen -disable-finalizer-msg` when creating each eSIMD module, uses
device allocations, and measures kernel time with Level Zero kernel timestamp
events. Its options mirror the SYCL paths:

```sh
./level_zero/peak_l0 mem --size-mib 1024 --iterations 100
./level_zero/peak_l0 compute --work-items 4096 --rounds 8192 --iterations 1000
```

Use `--spv FILE` with any mode to load a module from another path. The
runner asks Level Zero for the module's kernel names and falls back to the
known DPC++ dispatch-wrapper name because some drivers return an empty
kernel-name list for SPIR-V IL. `--kernel NAME` overrides that fallback when
loading a module built from a different declaration.

## Qwen3-8B greedy inference

The `qwen3/` subsystem runs batch-one BF16
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/main) inference on an
Intel BMG/B70 GPU. The host executable uses only C++17 and Level Zero; all
neural-network work is dispatched to the AOT eSIMD module. Activations and the
40,960-position grouped-query KV cache are BF16, while GEMV accumulation,
normalization statistics, attention/softmax, and reductions are FP32. Prompt
tokens deliberately reuse the cached single-token decode path.

Build the native engine and its BMG kernel image with oneAPI 2026:

```sh
make qwen3
make qwen3-test
make qwen3-gpu-test  # checkpoint-independent, requires a BMG GPU
```

The supported checkpoint is pinned to revision
`b968826d9c46dd6066d109eabc6255188de91218`. Downloading is explicit—the
engine and chat wrapper never access the network:

```sh
python3 -m pip install 'huggingface_hub[cli]'
huggingface-cli download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir /models/Qwen3-8B
```

Run the Level Zero engine on already-tokenized input. It writes only JSON to
stdout and diagnostics to stderr:

```sh
printf '151644 872 198' | ./qwen3_l0 \
  --model-dir /models/Qwen3-8B --input-ids - \
  --max-new-tokens 32 --max-seq-len 40960
```

For text chat, install the pinned reference/tokenizer dependencies and use the
Python wrapper. It applies the checkpoint's official Hugging Face chat
template and defaults to no-thinking greedy generation:

```sh
python3 -m pip install -r qwen3/requirements.txt
python3 qwen3/chat.py --model-dir /models/Qwen3-8B \
  --prompt 'Explain why the sky is blue in one sentence.' \
  --max-new-tokens 64 --no-thinking
```

`--thinking` enables the template's thinking mode, but decoding remains
greedy. The current milestone is single-turn and batch-one; sampling,
quantization, batching, parallel prefill, speculative decoding, and persisted
multi-turn caches are not implemented.

The engine validates the exact 36-layer architecture, all tensor names,
shapes, BF16 dtypes, shard bounds, input IDs, context length, BMG target, and
reported device memory before inference. It streams each tensor directly into
an aligned device arena, rather than retaining a checkpoint-sized host copy.
The full cache needs roughly 5.6 GiB in addition to the approximately 16 GiB
checkpoint and working memory.

Checkpoint-dependent conformance can use `qwen3/reference.py`, which pins
Transformers 4.51.0, eager attention, and BF16 weights. Set `MODEL_DIR` to the
pinned checkpoint when running local all-layer or generated-token comparisons;
the default `make qwen3-test` suite is synthetic and checkpoint-independent.

```sh
MODEL_DIR=/models/Qwen3-8B make qwen3-conformance
```
