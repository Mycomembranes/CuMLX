FC = gfortran
FFLAGS ?= -O3 -static-libgfortran -static-libgcc
CUDA_DRIVERKIT_LIB ?= ../../target/debug/libcuda_driverkit.dylib

PROGRAMS := nvidiactl_abi_smoke nvidiactl_utilization_smoke nvidiactl_core_control_probe cubin_launch_smoke nvgpu_smi_fortran fortran_gemm_smoke

.PHONY: all clean

all: $(PROGRAMS)

nvidiactl_abi_smoke: nvidiactl_abi_smoke.f90
	$(FC) $(FFLAGS) $< $(CUDA_DRIVERKIT_LIB) -o $@

nvidiactl_utilization_smoke: nvidiactl_utilization_smoke.f90
	$(FC) $(FFLAGS) $< $(CUDA_DRIVERKIT_LIB) -o $@

nvidiactl_core_control_probe: nvidiactl_core_control_probe.f90
	$(FC) $(FFLAGS) $< $(CUDA_DRIVERKIT_LIB) -o $@

cuda_driverkit_bindings.o: cuda_driverkit_bindings.f90
	$(FC) $(FFLAGS) -c $< -o $@

cubin_launch_smoke: cubin_launch_smoke.f90 cuda_driverkit_bindings.o
	$(FC) $(FFLAGS) $< cuda_driverkit_bindings.o $(CUDA_DRIVERKIT_LIB) -o $@

nvgpu_smi_fortran: nvgpu_smi_fortran.f90 cuda_driverkit_bindings.o
	$(FC) $(FFLAGS) $< cuda_driverkit_bindings.o $(CUDA_DRIVERKIT_LIB) -o $@

fortran_gemm_smoke: fortran_gemm_smoke.f90 cuda_driverkit_bindings.o
	$(FC) $(FFLAGS) $< cuda_driverkit_bindings.o $(CUDA_DRIVERKIT_LIB) -o $@

clean:
	rm -f $(PROGRAMS) *.o *.mod
