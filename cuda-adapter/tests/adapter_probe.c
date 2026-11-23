/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * adapter_probe.c - readiness gate for the launchers: can this process reach
 * an adapter server and does it report a device? Allocates nothing on the
 * device and launches nothing. Exit 0 = ready.
 */
#include <stdio.h>

int cuInit(unsigned int);
int cuDeviceGetCount(int *);
int cuDeviceGetName(char *, int, int);

int main(void) {
    char name[256] = "";
    int count = 0;
    int rc = cuInit(0);
    if (rc == 0) rc = cuDeviceGetCount(&count);
    if (rc == 0 && count > 0) rc = cuDeviceGetName(name, sizeof(name), 0);
    if (rc != 0 || count <= 0) {
        fprintf(stderr, "adapter_probe: not ready (CUDA error %d, %d device(s))\n", rc, count);
        return 3;
    }
    printf("adapter ready: %d device(s), device 0 = %s\n", count, name);
    return 0;
}
