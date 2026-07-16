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

.PHONY: all clean level-zero

all: peak_mem_2d peak_compute_dpas gemm gemv

gemm: gemm.cpp gemm_kernel.hpp
	$(CXX) $(CXXFLAGS) $(GEMM_SYCLFLAGS) $< -o $@

gemv: gemv.cpp gemv_kernel.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $< -o $@

peak_mem_2d: peak_mem_2d.cpp peak_mem_2d_kernel.cpp peak_mem_2d_kernel.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) peak_mem_2d.cpp \
		peak_mem_2d_kernel.cpp -o $@

peak_compute_dpas: peak_compute_dpas.cpp peak_compute_dpas_kernel.cpp \
	peak_compute_dpas_kernel.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) peak_compute_dpas.cpp \
		peak_compute_dpas_kernel.cpp -o $@

level-zero: level_zero/peak_l0 level_zero/peak_mem_2d.spv \
	level_zero/peak_compute_dpas.spv

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

clean:
	rm -f peak_mem_2d peak_compute_dpas gemm gemv level_zero/peak_l0 \
		level_zero/peak_mem_2d.spv level_zero/peak_compute_dpas.spv \
		level_zero/.peak_mem_spv_bundle level_zero/.peak_mem_image.* \
		level_zero/.peak_compute_spv_bundle \
		level_zero/.peak_compute_image.*
