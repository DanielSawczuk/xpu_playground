# XeTLA oneAPI 2026 compatibility patch

`xetla-oneapi-2026.patch` applies to the XeTLA revision pinned by
`checkout_source_material.sh`:

```text
intel/xetla@7a1acbde4ff608141500e142324923257605862a
```

The patch makes this archived XeTLA release compile with the oneAPI 2026.0
compiler. It includes:

- migration from legacy `<CL/sycl.hpp>`, `<ext/intel/esimd.hpp>`, and
  `cl::sycl` names to current `sycl/...` headers and the `sycl` namespace;
- compatibility fixes for current eSIMD and kernel APIs;
- standalone validation and profiling support for examples, removing their
  unconditional GoogleTest dependency;
- optional GoogleTest discovery for the test tree;
- updated example dispatch policies and Xe workgroup swizzles; and
- current event, exception, and profiling API usage throughout examples and
  tests.

## Apply

From the `xpu_playground` repository:

```sh
git -C ../xetla apply --check \
  "$PWD/patches/xetla-oneapi-2026.patch"
git -C ../xetla apply \
  "$PWD/patches/xetla-oneapi-2026.patch"
```

The patch expects a clean checkout at the pinned revision. To remove it before
making additional changes:

```sh
git -C ../xetla apply --reverse \
  "$PWD/patches/xetla-oneapi-2026.patch"
```

## Build

Build all examples without requiring GoogleTest:

```sh
cmake -S ../xetla -B ../xetla/build-oneapi-2026 \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF
cmake --build ../xetla/build-oneapi-2026 --parallel
```

To build tests too, install GoogleTest and configure with
`-DBUILD_TESTING=ON`. If GoogleTest is not found, CMake prints a warning and
continues building the examples.

## Validation

The patch was applied to a fresh detached worktree at the pinned commit and
all eleven example targets were built successfully with IntelLLVM/DPC++
2026.0.0. The compiler still reports upstream warnings about the deprecated
reserved `u1`/`s1` DPAS argument types and incomplete enum switches; these do
not prevent compilation.
