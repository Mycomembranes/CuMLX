/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * hybrid_ops_wrapper_abi.c - makes lib/drivers/cuda/cuda_compute.c loadable by
 * hybrid_ops/backends/cuda_backend.py.
 *
 * The two files disagree on five functions, so the Python backend cannot use
 * the C library as written (its no-argument cudac_get_device_name() call makes
 * the C side write through an uninitialised pointer). Neither file is edited.
 * The C source is compiled unchanged with the four clashing names renamed
 * (-Dcudac_x=cudac0_x), and this file provides the interface the Python
 * backend declares:
 *
 *   Python backend declares                        C source defines
 *   const char *cudac_get_device_name(void)        void (char *buf, int max_len)
 *   const char *cudac_get_last_error(void)         void (char *buf, int max_len)
 *   cudac_dispatch[_sync](pipeline,                (pipeline, handles, n,
 *       inputs, n, outputs, n, params, n,              grid x y z, block x y z)
 *       grid x y z)
 *   int cudac_synchronize(void)                    (not defined)
 *
 * The Python form has no block size. Its callers size the grid as
 * ceil(n / 256) and lib/drivers/cuda/kernels/cuda_kernels.h documents
 * "Block: (256,)", so the block is (256, 1, 1). That is this file's one
 * assumption; a kernel written for another block size would compute wrongly.
 */
#include <stdint.h>

#define ABI __attribute__((visibility("default")))
#define HYBRID_OPS_BLOCK_X 256

/* The C library's own functions (two of them under their renamed symbols). */
void cudac0_get_device_name(char *buf, int max_len);
void cudac0_get_last_error(char *buf, int max_len);
int cudac_dispatch_ext(uint64_t pipeline, const uint64_t *read_handles, int num_read,
                       const uint64_t *write_handles, int num_write, const uint64_t *param_handles,
                       int num_params, int grid_x, int grid_y, int grid_z, int block_x, int block_y,
                       int block_z);
int cudaDeviceSynchronize(void);

ABI const char *cudac_get_device_name(void) {
    static char name[256];
    cudac0_get_device_name(name, (int)sizeof(name));
    return name;
}

/* NULL when there is no error text: the Python side prints "unknown" then. */
ABI const char *cudac_get_last_error(void) {
    static _Thread_local char text[512];
    text[0] = 0;
    cudac0_get_last_error(text, (int)sizeof(text));
    return text[0] ? text : (const char *)0;
}

/* cudac_dispatch_ext launches on the open batch if there is one, and otherwise
 * launches and waits, which is what the two Python entry points need. */
static int dispatch(uint64_t pipeline, const uint64_t *inputs, uint32_t num_inputs, const uint64_t *outputs,
                    uint32_t num_outputs, const uint64_t *params, uint32_t num_params, uint32_t grid_x,
                    uint32_t grid_y, uint32_t grid_z) {
    return cudac_dispatch_ext(pipeline, inputs, (int)num_inputs, outputs, (int)num_outputs, params,
                              (int)num_params, (int)grid_x, (int)grid_y, (int)grid_z, HYBRID_OPS_BLOCK_X, 1, 1);
}

ABI int cudac_dispatch(uint64_t pipeline, const uint64_t *inputs, uint32_t num_inputs, const uint64_t *outputs,
                       uint32_t num_outputs, const uint64_t *params, uint32_t num_params, uint32_t grid_x,
                       uint32_t grid_y, uint32_t grid_z) {
    return dispatch(pipeline, inputs, num_inputs, outputs, num_outputs, params, num_params, grid_x, grid_y, grid_z);
}

ABI int cudac_dispatch_sync(uint64_t pipeline, const uint64_t *inputs, uint32_t num_inputs,
                            const uint64_t *outputs, uint32_t num_outputs, const uint64_t *params,
                            uint32_t num_params, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z) {
    return dispatch(pipeline, inputs, num_inputs, outputs, num_outputs, params, num_params, grid_x, grid_y, grid_z);
}

ABI int cudac_synchronize(void) { return cudaDeviceSynchronize() == 0 ? 0 : -1; }
