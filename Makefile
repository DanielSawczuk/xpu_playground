CXX = icpx
DEVICE ?= bmg
CXXFLAGS ?= -O3 -std=c++17
SYCLFLAGS ?= -fsycl -fsycl-targets=spir64_gen \
	-Xsycl-target-backend=spir64_gen "-device $(DEVICE)" \
	-fsycl-device-code-split=per_kernel
GEMM_SYCLFLAGS ?= -fsycl -fsycl-targets=spir64_gen \
	-Xsycl-target-backend=spir64_gen \
	"-device $(DEVICE) -options '-doubleGRF'" \
	-fsycl-device-code-split=per_kernel

DPCPP ?= dpcpp
HOST_CXX ?= c++
DPCPP_BIN_DIR := $(dir $(shell command -v $(DPCPP)))
OFFLOAD_EXTRACT ?= $(DPCPP_BIN_DIR)compiler/clang-offload-extract
SPIRVFLAGS ?= -O3 -std=c++17 -fsycl -fsycl-targets=spir64 \
	-fsycl-device-code-split=per_kernel
AOTFLAGS ?= $(CXXFLAGS) $(SYCLFLAGS)
GEMM_AOTFLAGS ?= $(CXXFLAGS) $(GEMM_SYCLFLAGS)
ONEDNN_ROOT ?= /opt/intel/oneapi/dnnl/latest
ONEDNN_FLAGS ?= -I$(ONEDNN_ROOT)/include -L$(ONEDNN_ROOT)/lib \
	-Wl,--disable-new-dtags,-rpath,$(ONEDNN_ROOT)/lib -ldnnl

.PHONY: all clean level-zero onednn qwen3 qwen3-kernels qwen3-test \
	qwen3-gpu-test qwen3-conformance

all: peak_sycl peak_mem_2d peak_compute_dpas gemm gemv

onednn: onednn_sycl level_zero/onednn_l0

onednn_sycl: onednn_sycl.cpp onednn_matmul.hpp
	$(CXX) $(CXXFLAGS) -fsycl $< $(ONEDNN_FLAGS) -o $@

level_zero/onednn_l0: level_zero/onednn_l0.cpp onednn_matmul.hpp
	$(HOST_CXX) $(CXXFLAGS) $< $(ONEDNN_FLAGS) -lze_loader -o $@

peak_sycl: peak_sycl.cpp peak_mem_2d peak_compute_dpas gemv gemm
	$(HOST_CXX) $(CXXFLAGS) peak_sycl.cpp -o $@

gemm: gemm.cpp gemm_kernel.hpp
	$(CXX) $(CXXFLAGS) -DXPU_PLAYGROUND_STANDALONE_HOST \
		$(GEMM_SYCLFLAGS) $< -o $@

gemv: gemv.cpp gemv_kernel.hpp
	$(CXX) $(CXXFLAGS) -DXPU_PLAYGROUND_STANDALONE_HOST $(SYCLFLAGS) $< -o $@

peak_mem_2d: peak_mem_2d.cpp peak_mem_2d_kernel.cpp peak_mem_2d_kernel.hpp
	$(CXX) $(CXXFLAGS) -DXPU_PLAYGROUND_STANDALONE_HOST $(SYCLFLAGS) \
		peak_mem_2d.cpp \
		peak_mem_2d_kernel.cpp -o $@

peak_compute_dpas: peak_compute_dpas.cpp peak_compute_dpas_kernel.cpp \
	peak_compute_dpas_kernel.hpp
	$(CXX) $(CXXFLAGS) -DXPU_PLAYGROUND_STANDALONE_HOST $(SYCLFLAGS) \
		peak_compute_dpas.cpp \
		peak_compute_dpas_kernel.cpp -o $@

GEM_MODULES := level_zero/gemv_bf16.bin level_zero/gemv_fp16.bin \
	level_zero/gemm_bf16_small.bin level_zero/gemm_bf16_large.bin \
	level_zero/gemm_fp16_small.bin level_zero/gemm_fp16_large.bin

level-zero: level_zero/peak_l0 level_zero/peak_mem_2d.spv \
	level_zero/peak_compute_dpas.spv $(GEM_MODULES)

qwen3: qwen3_l0 qwen3_kernels.bin

qwen3-kernels: qwen3_kernels.bin

qwen3_l0: qwen3/main.cpp qwen3/model.cpp qwen3/model.hpp \
	qwen3/level_zero_runtime.cpp qwen3/level_zero_runtime.hpp
	$(HOST_CXX) -O3 -std=c++17 qwen3/main.cpp qwen3/model.cpp \
		qwen3/level_zero_runtime.cpp -lze_loader -o $@

qwen3_kernels.bin: qwen3/kernels_entry.cpp qwen3/kernels.hpp gemv_kernel.hpp
	$(DPCPP) -O3 -std=c++17 -fsycl -fsycl-targets=spir64_gen \
		-Xsycl-target-backend=spir64_gen "-device $(DEVICE)" \
		-fsycl-device-code-split=off qwen3/kernels_entry.cpp \
		-o qwen3/.qwen3_kernel_bundle
	$(OFFLOAD_EXTRACT) -q --stem=qwen3/.qwen3_kernel_image \
		qwen3/.qwen3_kernel_bundle
	mv qwen3/.qwen3_kernel_image.0 $@
	rm -f qwen3/.qwen3_kernel_bundle qwen3/.qwen3_kernel_image.*

qwen3/tests/test_cpu_reference: qwen3/tests/test_cpu_reference.cpp \
	qwen3/model.cpp qwen3/model.hpp
	$(HOST_CXX) -O2 -std=c++17 -Wall -Wextra -Werror \
		qwen3/tests/test_cpu_reference.cpp qwen3/model.cpp -o $@

qwen3-test: qwen3/tests/test_cpu_reference
	./qwen3/tests/test_cpu_reference
	python3 -m py_compile qwen3/chat.py qwen3/reference.py \
		qwen3/tests/test_conformance.py

qwen3/tests/test_gpu_dispatch: qwen3/tests/test_gpu_dispatch.cpp \
	qwen3/model.cpp qwen3/model.hpp qwen3/level_zero_runtime.cpp \
	qwen3/level_zero_runtime.hpp
	$(HOST_CXX) -O2 -std=c++17 -Wall -Wextra -Werror \
		qwen3/tests/test_gpu_dispatch.cpp qwen3/model.cpp \
		qwen3/level_zero_runtime.cpp -lze_loader -o $@

qwen3-gpu-test: qwen3_kernels.bin qwen3/tests/test_gpu_dispatch
	./qwen3/tests/test_gpu_dispatch ./qwen3_kernels.bin

qwen3-conformance: qwen3
	python3 -m unittest qwen3.tests.test_conformance

level_zero/peak_l0: level_zero/peak_l0.cpp
	$(HOST_CXX) -O3 -std=c++17 $< -lze_loader -o $@

level_zero/peak_mem_2d.spv: level_zero/peak_mem_spv_entry.cpp \
	peak_mem_2d_kernel.cpp peak_mem_2d_kernel.hpp
	$(DPCPP) $(SPIRVFLAGS) level_zero/peak_mem_spv_entry.cpp \
		peak_mem_2d_kernel.cpp \
		-o level_zero/.peak_mem_spv_bundle
	cd level_zero && $(OFFLOAD_EXTRACT) -q --stem=.peak_mem_image \
		.peak_mem_spv_bundle
	@set -e; found=0; for image in level_zero/.peak_mem_image.*; do \
		if strings "$$image" | grep -Fq \
		'_ZTSN12_GLOBAL__N_121PeakMemLevelZeroEntryE'; then \
			mv "$$image" $@; found=1; \
		fi; \
	done; \
	rm -f level_zero/.peak_mem_spv_bundle \
		level_zero/.peak_mem_image.*; \
	test $$found -eq 1

level_zero/peak_compute_dpas.spv: level_zero/peak_compute_spv_entry.cpp \
	peak_compute_dpas_kernel.cpp peak_compute_dpas_kernel.hpp
	$(DPCPP) $(SPIRVFLAGS) level_zero/peak_compute_spv_entry.cpp \
		peak_compute_dpas_kernel.cpp \
		-o level_zero/.peak_compute_spv_bundle
	cd level_zero && $(OFFLOAD_EXTRACT) -q --stem=.peak_compute_image \
		.peak_compute_spv_bundle
	@set -e; found=0; for image in level_zero/.peak_compute_image.*; do \
		if strings "$$image" | grep -Fq \
		'_ZTSN12_GLOBAL__N_125PeakComputeLevelZeroEntryE'; then \
			mv "$$image" $@; found=1; \
		fi; \
	done; \
	rm -f level_zero/.peak_compute_spv_bundle \
		level_zero/.peak_compute_image.*; \
	test $$found -eq 1

define make_gem_module
level_zero/$(1).bin: level_zero/gem_spv_entry.cpp gemv_kernel.hpp gemm_kernel.hpp
	$$(DPCPP) $$($(3)) -D$(2) level_zero/gem_spv_entry.cpp \
		-o level_zero/.$(1)_bundle
	cd level_zero && $$(OFFLOAD_EXTRACT) -q --stem=.$(1)_image \
		.$(1)_bundle
	mv level_zero/.$(1)_image.0 $$@
	rm -f level_zero/.$(1)_bundle level_zero/.$(1)_image.*
endef

$(eval $(call make_gem_module,gemv_bf16,GEMV_BF16,AOTFLAGS))
$(eval $(call make_gem_module,gemv_fp16,GEMV_FP16,AOTFLAGS))
$(eval $(call make_gem_module,gemm_bf16_small,GEMM_BF16_SMALL,GEMM_AOTFLAGS))
$(eval $(call make_gem_module,gemm_bf16_large,GEMM_BF16_LARGE,GEMM_AOTFLAGS))
$(eval $(call make_gem_module,gemm_fp16_small,GEMM_FP16_SMALL,GEMM_AOTFLAGS))
$(eval $(call make_gem_module,gemm_fp16_large,GEMM_FP16_LARGE,GEMM_AOTFLAGS))

clean:
	rm -f peak_sycl peak_mem_2d peak_compute_dpas gemm gemv level_zero/peak_l0 \
		onednn_sycl level_zero/onednn_l0 \
		level_zero/peak_mem_2d.spv level_zero/peak_compute_dpas.spv \
		$(GEM_MODULES) level_zero/.*_bundle level_zero/.*_image.* \
		level_zero/.peak_mem_spv_bundle level_zero/.peak_mem_image.* \
		level_zero/.peak_compute_spv_bundle \
		level_zero/.peak_compute_image.* qwen3_l0 qwen3_kernels.bin \
		qwen3/.qwen3_kernel_bundle qwen3/.qwen3_kernel_image.* \
		qwen3/tests/test_cpu_reference qwen3/tests/test_gpu_dispatch
