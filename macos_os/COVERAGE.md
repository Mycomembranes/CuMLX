# macOS OS layer coverage (OGK 570.190)

What `shims/macos_os/` provides in place of the Linux OS-layer files, generated from the sources on 2026-11-20. It covers every `os_*` / `nv_*` symbol that the RM closure (`closure-570.190.txt`) references from `os-interface.c`, `os-pci.c`, `os-mlock.c`, `os-registry.c` and `nv.c`, plus the `nv_*`/`libspdm_*` entry points of the Linux-only files the closure cannot compile (`nv-dma.c`, `nv-acpi.c`, `nv-nano-timer.c`, `libspdm_aead.c`, `libspdm_hmac_sha.c`), plus `os_pci_trigger_flr`, `os_lock_user_pages` and `os_lookup_user_io_memory`, which are refused explicitly.

**Totals (OS-layer rows, before the shim/thread rows at the end): 257 entries. 138 implemented (18 of them data), 112 refused, 7 stub.** The six timer rows moved from refused to implemented on 2026-11-20.

- **implemented**: real userspace behaviour (pthread, nanosleep, clock_gettime, malloc), or the device path shown.
- **refused**: prints one stderr note per function and returns `NV_ERR_NOT_SUPPORTED`, or for a `void`/`bool`/pointer function the "unavailable" value (`NULL`, `NV_FALSE`, -1). It never calls the KPI; `tests/test_macos_os.c` checks this through the mock call counter.
- **stub**: answers but does not do the real thing; the row says how.

What is enforced (2026-11-20 revision; the closure link uses the `KOT_SAFE_SHIMS` builds of `shims/shim_{io,pci,dma,firmware}.c`, never the default-build objects, which keep the legacy unguarded code byte-identical):

- **Identity** (`kot_device.c`): no config, MMIO or DMA access until `KOT_REGISTRY_ENTRY_ID` (16 hex digits) equals the registry entry id the KPI reports, with exactly one NVIDIA device present. Refusal is sticky.
- **PCI config**: one choke point, `kot_cfg_read` / `kot_cfg_write`, used by `os_pci_macos.c` and `shims/shim_pci.c`: compiled-in write policy (`kot_write_policy.h`: deny command, BARs, ROM, PMCSR, MSI/MSI-X control, PCIe Device/Link Control, extended config; deny all if the capability walk fails), `shim_mutation_admit` (really on: `SHIM_MUTATION_PREGATE` is defined for every `KOT_SAFE_SHIMS` compile and its absence is a build error), aligned dword read-modify-write for byte/word writes with no write-back after a failed read, ring log of issued writes.
- **BAR MMIO**: registry tokens (`kot_tokens.h`); `shim_read*`/`shim_write*` validate, bounds-check (mapping and KPI BAR size), apply the MMIO write policy (deny `NV_PMC_ENABLE`, `NV_PMC_DEVICE_ENABLE`, GSP/SEC2/FSP falcon ENGINE resets, the BAR0 config mirror) and `shim_mutation_admit`, log every issued write. `KOT_WRITE_ALLOWLIST` lifts individual denies.
- **DMA**: `dma_free_coherent` (and `nv_free_pages` through it) quarantines; `cudaDriverKitKpiDmaFree` only after `kot_device_quiesced_set()` (`kot_lifecycle.h` says what must be proven first). Double free and foreign pointers are reported and counted.
- **Fatal paths**: trap stubs, `os_bug_check`, the fatal-signal handler and a process exit with live/quarantined DMA all go through `kot_fatal`, which parks the thread while device state may be owned (`KOT_FATAL_ABORTS=1` for the mock harness and tests only).
- **Firmware**: absolute `$NVIDIA_KO_FIRMWARE_ROOT`, no aliases, only the four 570.190 files with their SHA256 pins.
- Nothing here opens a device, a socket or a file other than the pinned firmware and the optional allowlist. Interrupts: polling model (`macos_os_priv.h`); the RM register path is routed to `shim_readl`/`shim_writel` by `override/nv-priv.h`. Registry values come from `KOT_RM_REGISTRY` (see `os_registry_macos.c`).

| function | Linux source | status | device path |
|---|---|---|---|
| `os_page_size` | os-interface.c | implemented (data) |  |
| `os_page_mask` | os-interface.c | implemented (data) |  |
| `os_page_shift` | os-interface.c | implemented (data) |  |
| `os_cc_enabled` | os-interface.c | implemented (data) |  |
| `os_cc_sev_snp_enabled` | os-interface.c | implemented (data) |  |
| `os_cc_snp_vtom_enabled` | os-interface.c | implemented (data) |  |
| `os_cc_tdx_enabled` | os-interface.c | implemented (data) |  |
| `os_dma_buf_enabled` | os-interface.c | implemented (data) |  |
| `os_imex_channel_is_supported` | os-interface.c | implemented (data) |  |
| `os_is_isr` | os-interface.c | implemented |  |
| `os_semaphore_may_sleep` | os-interface.c | implemented |  |
| `os_alloc_mem` | os-interface.c | implemented |  |
| `os_free_mem` | os-interface.c | implemented |  |
| `os_mem_copy` | os-interface.c | implemented |  |
| `os_mem_set` | os-interface.c | implemented |  |
| `os_mem_cmp` | os-interface.c | implemented |  |
| `os_memcpy_from_user` | os-interface.c | implemented |  |
| `os_memcpy_to_user` | os-interface.c | implemented |  |
| `os_string_compare` | os-interface.c | implemented |  |
| `os_string_copy` | os-interface.c | implemented |  |
| `os_string_length` | os-interface.c | implemented |  |
| `os_strtoul` | os-interface.c | implemented |  |
| `os_snprintf` | os-interface.c | implemented |  |
| `os_get_current_time` | os-interface.c | implemented |  |
| `os_get_current_tick` | os-interface.c | implemented |  |
| `os_get_current_tick_hr` | os-interface.c | implemented |  |
| `os_get_tick_resolution` | os-interface.c | implemented |  |
| `os_delay_us` | os-interface.c | implemented |  |
| `os_delay` | os-interface.c | implemented |  |
| `os_schedule` | os-interface.c | implemented |  |
| `os_get_current_process` | os-interface.c | implemented |  |
| `os_get_current_process_name` | os-interface.c | implemented |  |
| `os_get_current_thread` | os-interface.c | implemented |  |
| `os_get_euid` | os-interface.c | implemented |  |
| `os_is_administrator` | os-interface.c | implemented |  |
| `os_get_cpu_count` | os-interface.c | implemented |  |
| `os_get_cpu_number` | os-interface.c | stub: returns 0 (macOS has no getcpu) |  |
| `os_get_cpu_frequency` | os-interface.c | implemented |  |
| `os_get_max_user_va` | os-interface.c | implemented |  |
| `os_get_is_openrm` | os-interface.c | implemented |  |
| `os_get_version_info` | os-interface.c | implemented |  |
| `os_get_random_bytes` | os-interface.c | implemented |  |
| `os_is_init_ns` | os-interface.c | implemented |  |
| `os_is_nvswitch_present` | os-interface.c | implemented |  |
| `os_device_vm_present` | os-interface.c | implemented |  |
| `os_imex_channel_count` | os-interface.c | implemented |  |
| `nv_printf` | os-interface.c | implemented |  |
| `out_string` | os-interface.c | implemented |  |
| `os_dbg_init` | os-interface.c | implemented |  |
| `os_dbg_set_level` | os-interface.c | implemented |  |
| `os_dbg_breakpoint` | os-interface.c | implemented |  |
| `os_dump_stack` | os-interface.c | implemented |  |
| `os_bug_check` | os-interface.c | implemented: prints, dumps the stack, then `kot_fatal` (parks; aborts only if nothing device-related was touched or `KOT_FATAL_ABORTS=1`) |  |
| `os_alloc_mutex` | os-interface.c | implemented |  |
| `os_free_mutex` | os-interface.c | implemented |  |
| `os_acquire_mutex` | os-interface.c | implemented |  |
| `os_cond_acquire_mutex` | os-interface.c | implemented |  |
| `os_release_mutex` | os-interface.c | implemented |  |
| `os_alloc_semaphore` | os-interface.c | implemented |  |
| `os_free_semaphore` | os-interface.c | implemented |  |
| `os_acquire_semaphore` | os-interface.c | implemented |  |
| `os_cond_acquire_semaphore` | os-interface.c | implemented |  |
| `os_release_semaphore` | os-interface.c | implemented |  |
| `os_alloc_rwlock` | os-interface.c | implemented |  |
| `os_free_rwlock` | os-interface.c | implemented |  |
| `os_acquire_rwlock_read` | os-interface.c | implemented |  |
| `os_acquire_rwlock_write` | os-interface.c | implemented |  |
| `os_cond_acquire_rwlock_read` | os-interface.c | implemented |  |
| `os_cond_acquire_rwlock_write` | os-interface.c | implemented |  |
| `os_release_rwlock_read` | os-interface.c | implemented |  |
| `os_release_rwlock_write` | os-interface.c | implemented |  |
| `os_alloc_spinlock` | os-interface.c | implemented |  |
| `os_free_spinlock` | os-interface.c | implemented |  |
| `os_acquire_spinlock` | os-interface.c | implemented: pthread mutex inside the interrupt-exclusion gate (a holder has "interrupts off"; the polled ISR waits for all holders) |  |
| `os_release_spinlock` | os-interface.c | implemented |  |
| `os_alloc_wait_queue` | os-interface.c | implemented |  |
| `os_free_wait_queue` | os-interface.c | implemented |  |
| `os_wait_uninterruptible` | os-interface.c | implemented |  |
| `os_wait_interruptible` | os-interface.c | implemented |  |
| `os_wake_up` | os-interface.c | implemented |  |
| `os_queue_work_item` | os-interface.c | implemented: single FIFO worker, `rm_execute_work_item(NULL, data)` (arm64 `NV_ENTER_RM_RUNTIME` ignores the stack); refused (`NV_ERR_NOT_READY`) after `kot_work_queue_stop` or `kot_fatal` |  |
| `os_flush_work_queue` | os-interface.c | implemented: waits for the items queued before the call (sequence numbers), not for items queued later |  |
| `os_is_queue_flush_ongoing` | os-interface.c | implemented |  |
| `os_map_kernel_space` | os-interface.c | implemented: NULL unless the range is inside a BAR of the accepted device | shim_io.c ioremap() registry token; shim_read*/shim_write* validate it, then cudaDriverKitKpiReadReg32/WriteReg32 |
| `os_unmap_kernel_space` | os-interface.c | implemented: the token becomes stale (never reused) | shim_io.c iounmap() (no KPI call) |
| `os_flush_cpu_write_combine_buffer` | os-interface.c | implemented |  |
| `os_flush_cpu_cache_all` | os-interface.c | refused |  |
| `os_flush_user_cache` | os-interface.c | refused |  |
| `os_alloc_pages_node` | os-interface.c | refused |  |
| `os_get_page` | os-interface.c | refused |  |
| `os_put_page` | os-interface.c | refused |  |
| `os_get_page_refcount` | os-interface.c | refused |  |
| `os_count_tail_pages` | os-interface.c | refused |  |
| `os_offline_page_at_address` | os-interface.c | refused |  |
| `os_numa_memblock_size` | os-interface.c | refused |  |
| `os_numa_add_gpu_memory` | os-interface.c | refused |  |
| `os_numa_remove_gpu_memory` | os-interface.c | refused |  |
| `os_get_numa_node_memory_usage` | os-interface.c | refused |  |
| `os_add_record_for_crashLog` | os-interface.c | refused |  |
| `os_delete_record_for_crashLog` | os-interface.c | refused |  |
| `os_get_acpi_rsdp_from_uefi` | os-interface.c | refused |  |
| `os_get_smbios_header` | os-interface.c | refused |  |
| `os_is_efi_enabled` | os-interface.c | refused |  |
| `os_is_grid_supported` | os-interface.c | implemented |  |
| `os_get_grid_csp_support` | os-interface.c | implemented |  |
| `os_is_vgx_hyper` | os-interface.c | implemented |  |
| `os_call_vgpu_vfio` | os-interface.c | refused |  |
| `os_inject_vgx_msi` | os-interface.c | refused |  |
| `os_disable_console_access` | os-interface.c | refused |  |
| `os_enable_console_access` | os-interface.c | refused |  |
| `os_open_temporary_file` | os-interface.c | refused |  |
| `os_close_file` | os-interface.c | refused |  |
| `os_write_file` | os-interface.c | refused |  |
| `os_read_file` | os-interface.c | refused |  |
| `os_nv_cap_create_dir_entry` | os-interface.c | refused |  |
| `os_nv_cap_create_file_entry` | os-interface.c | refused |  |
| `os_nv_cap_destroy_entry` | os-interface.c | refused |  |
| `os_nv_cap_validate_and_dup_fd` | os-interface.c | refused |  |
| `os_nv_cap_close_fd` | os-interface.c | refused |  |
| `os_check_access` | os-interface.c | refused |  |
| `os_find_ns_pid` | os-interface.c | refused |  |
| `os_get_pid_info` | os-interface.c | refused |  |
| `os_put_pid_info` | os-interface.c | stub: no-op (os_get_pid_info is refused) |  |
| `os_io_read_byte` | os-interface.c | refused |  |
| `os_io_read_word` | os-interface.c | refused |  |
| `os_io_write_byte` | os-interface.c | refused |  |
| `os_io_write_word` | os-interface.c | refused |  |
| `os_enable_pci_req_atomics` | os-pci.c | refused |  |
| `os_imex_channel_get` | os-interface.c | refused |  |
| `os_pci_init_handle` | os-pci.c | implemented: NULL unless the identity gate accepts the device | kot_device_ready: cudaDriverKitKpiDeviceCount + ResolveDevice (read-only) |
| `os_pci_read_byte` | os-pci.c | implemented | kot_cfg_read -> cudaDriverKitKpiPciReadConfig (dword) |
| `os_pci_read_word` | os-pci.c | implemented | kot_cfg_read -> cudaDriverKitKpiPciReadConfig (dword) |
| `os_pci_read_dword` | os-pci.c | implemented | kot_cfg_read -> cudaDriverKitKpiPciReadConfig (dword) |
| `os_pci_write_byte` | os-pci.c | implemented (denied writes: `NV_ERR_NOT_SUPPORTED`) | kot_cfg_write: write policy + shim_mutation_admit + cudaDriverKitKpiPciReadConfig (dword RMW, no write after a failed read) + cudaDriverKitKpiPciWriteConfig + ring log |
| `os_pci_write_word` | os-pci.c | implemented (denied writes: `NV_ERR_NOT_SUPPORTED`) | kot_cfg_write: write policy + shim_mutation_admit + cudaDriverKitKpiPciReadConfig (dword RMW, no write after a failed read) + cudaDriverKitKpiPciWriteConfig + ring log |
| `os_pci_write_dword` | os-pci.c | implemented (denied writes: `NV_ERR_NOT_SUPPORTED`) | kot_cfg_write: write policy + shim_mutation_admit + cudaDriverKitKpiPciWriteConfig + ring log |
| `os_pci_remove_supported` | os-pci.c | implemented |  |
| `os_pci_remove` | os-pci.c | refused |  |
| `os_pci_trigger_flr` | os-pci.c | refused |  |
| `os_lookup_user_io_memory` | os-mlock.c | refused |  |
| `os_lock_user_pages` | os-mlock.c | refused |  |
| `os_unlock_user_pages` | os-mlock.c | refused |  |
| `os_registry_init` | os-registry.c | implemented |  |
| `nv_alloc_pages` | nv.c | implemented (refuses `unencrypted` CC pages) | shim_dma.c dma_alloc_coherent -> cudaDriverKitKpiDmaAlloc |
| `nv_free_pages` | nv.c | implemented: registry-checked (double free / foreign pointer reported, `NV_ERR_INVALID_ARGUMENT`) | shim_dma.c dma_free_coherent: quarantine; cudaDriverKitKpiDmaFree only once quiesced |
| `nv_alloc_kernel_mapping` | nv.c | implemented | no KPI: CPU pointer into the KPI DMA buffer |
| `nv_free_kernel_mapping` | nv.c | implemented |  |
| `nv_dma_map_alloc` | nv-dma.c | implemented |  |
| `nv_dma_unmap_alloc` | nv-dma.c | implemented |  |
| `nv_dma_cache_invalidate` | nv-dma.c | implemented | shim_dma.c dma_sync_single_for_cpu -> cudaDriverKitKpiDmaSyncFromDevice |
| `nv_dma_map_mmio` | nv-dma.c | refused |  |
| `nv_dma_unmap_mmio` | nv-dma.c | refused |  |
| `nv_dma_map_peer` | nv-dma.c | refused |  |
| `nv_dma_unmap_peer` | nv-dma.c | refused |  |
| `nv_dma_import_sgt` | nv-dma.c | refused |  |
| `nv_dma_release_sgt` | nv-dma.c | refused |  |
| `nv_flush_coherent_cpu_cache_range` | nv.c | stub: barrier only (no userspace range clean/invalidate) |  |
| `nv_get_firmware` | nv.c | implemented | no KPI: shim_firmware.c request_firmware under $NVIDIA_KO_FIRMWARE_ROOT |
| `nv_put_firmware` | nv.c | implemented | no KPI: shim_firmware.c release_firmware |
| `nv_get_ctl_state` | nv.c | implemented |  |
| `nv_is_gpu_accessible` | nv.c | implemented | cudaDriverKitKpiProbe (read-only) |
| `nv_set_dma_address_size` | nv.c | implemented |  |
| `nv_requires_dma_remap` | nv.c | implemented |  |
| `nv_get_dev_minor` | nv.c | stub: returns 0 (one device) |  |
| `nv_is_chassis_notebook` | nv.c | implemented |  |
| `nv_platform_supports_s0ix` | nv.c | implemented |  |
| `nv_s2idle_pm_configured` | nv.c | implemented |  |
| `nv_dynamic_power_available` | nv.c | implemented |  |
| `nv_log_error` | nv.c | implemented |  |
| `nv_post_event` | nv.c | refused |  |
| `nv_get_event` | nv.c | refused |  |
| `nv_get_file_private` | nv.c | refused |  |
| `nv_put_file_private` | nv.c | stub: no-op (nv_get_file_private is refused) |  |
| `nv_allow_runtime_suspend` | nv.c | refused |  |
| `nv_disallow_runtime_suspend` | nv.c | refused |  |
| `nv_idle_holdoff` | nv.c | refused |  |
| `nv_indicate_idle` | nv.c | refused |  |
| `nv_indicate_not_idle` | nv.c | refused |  |
| `nv_audio_dynamic_power` | nv.c | refused |  |
| `nv_control_soc_irqs` | nv.c | refused |  |
| `nv_get_screen_info` | nv.c | refused |  |
| `nv_set_primary_vga_status` | nv.c | refused |  |
| `nv_get_disp_smmu_stream_ids` | nv.c | refused |  |
| `nv_match_gpu_os_info` | nv.c | refused |  |
| `nv_get_device_memory_config` | nv.c | refused |  |
| `nv_get_egm_info` | nv.c | refused |  |
| `nv_start_rc_timer` | nv.c | implemented (kot_timer_macos.c): 1 s timer, `rm_run_rc_callback(NULL, nv)` on the worker, re-armed while it returns `NV_OK` |  |
| `nv_stop_rc_timer` | nv.c | implemented: clears `rc_timer_enabled`, cancels (waits for a running callback unless on the worker) |  |
| `nv_create_nano_timer` | nv-nano-timer.c | implemented (kot_timer_macos.c): one timer thread, monotonic clock |  |
| `nv_start_nano_timer` | nv-nano-timer.c | implemented: expiry posted to the worker, `rm_run_nano_timer_callback(NULL, nv, pTmrEvent)` |  |
| `nv_cancel_nano_timer` | nv-nano-timer.c | implemented: pending expiry dropped (generation check) |  |
| `nv_destroy_nano_timer` | nv-nano-timer.c | implemented: cancelled and marked destroyed; kept allocated (a posted expiry may still refer to it) |  |
| `nv_schedule_uvm_isr` | nv.c | refused |  |
| `nv_schedule_uvm_drain_p2p` | nv.c | refused |  |
| `nv_schedule_uvm_resume_p2p` | nv.c | refused |  |
| `nv_register_user_pages` | nv.c | refused |  |
| `nv_unregister_user_pages` | nv.c | refused |  |
| `nv_register_peer_io_mem` | nv.c | refused |  |
| `nv_unregister_peer_io_mem` | nv.c | refused |  |
| `nv_register_phys_pages` | nv.c | refused |  |
| `nv_unregister_phys_pages` | nv.c | refused |  |
| `nv_register_sgt` | nv.c | refused |  |
| `nv_unregister_sgt` | nv.c | refused |  |
| `nv_alias_pages` | nv.c | refused |  |
| `nv_get_num_phys_pages` | nv.c | refused |  |
| `nv_get_phys_pages` | nv.c | refused |  |
| `nv_acquire_fabric_mgmt_cap` | nv.c | refused |  |
| `nv_acpi_ddc_method` | nv-acpi.c | refused |  |
| `nv_acpi_dod_method` | nv-acpi.c | refused |  |
| `nv_acpi_dsm_method` | nv-acpi.c | refused |  |
| `nv_acpi_mux_method` | nv-acpi.c | refused |  |
| `nv_acpi_rom_method` | nv-acpi.c | refused |  |
| `nv_acpi_get_powersource` | nv-acpi.c | refused |  |
| `nv_acpi_is_battery_present` | nv-acpi.c | refused |  |
| `nv_acpi_methods_init` | nv-acpi.c | refused |  |
| `nv_acpi_methods_uninit` | nv-acpi.c | refused |  |
| `libspdm_aead_gcm_prealloc` | libspdm_aead.c | refused |  |
| `libspdm_aead_free` | libspdm_aead.c | refused |  |
| `libspdm_aead_aes_gcm_encrypt_prealloc` | libspdm_aead.c | refused |  |
| `libspdm_aead_aes_gcm_decrypt_prealloc` | libspdm_aead.c | refused |  |
| `libspdm_hmac_sha256_new` | libspdm_hmac_sha.c | refused |  |
| `libspdm_hmac_sha256_free` | libspdm_hmac_sha.c | refused |  |
| `libspdm_hmac_sha256_set_key` | libspdm_hmac_sha.c | refused |  |
| `libspdm_hmac_sha256_update` | libspdm_hmac_sha.c | refused |  |
| `libspdm_hmac_sha256_final` | libspdm_hmac_sha.c | refused |  |
| `nvidia_caps_root` | nv.c | implemented (data) |  |
| `NVreg_RegisterPCIDriver` | nv-reg.h (macro) | implemented (data) |  |
| `NVreg_GrdmaPciTopoCheckOverride` | nv-reg.h (macro) | implemented (data) |  |
| `num_nv_devices` | nv.c | implemented (data) |  |
| `num_probed_nv_devices` | nv.c | implemented (data) |  |
| `nv_ats_supported` | nv.c | implemented (data) |  |
| `nv_device_name` | nv.c | implemented (data) |  |
| `nv_linux_devices` | nv.c | implemented (data) |  |
| `nv_linux_devices_lock` | nv.c | implemented (data) |  |
| `find_uuid` | nv.c | stub: returns NULL (no nv_linux_state_t list) |  |
| `nv_is_uuid_in_gpu_exclusion_list` | os-registry.c | implemented |  |
| `nv_linux_add_device_locked` | nv.c | refused |  |
| `nv_linux_remove_device_locked` | nv.c | refused |  |
| `nv_linux_init_open_q` | nv.c | refused |  |
| `nv_linux_stop_open_q` | nv.c | refused |  |
| `nv_lock_init_locks` | nv.c | refused |  |
| `nv_lock_destroy_locks` | nv.c | refused |  |
| `nv_dev_free_stacks` | nv.c | refused |  |
| `nv_parse_per_device_option_string` | os-registry.c | refused |  |
| `nv_shutdown_adapter` | nv.c | refused |  |
| `nv_acpi_unregister_notifier` | nv-acpi.c | refused |  |
| `nv_acpi_power_resource_method_present` | nv-acpi.c | refused |  |
| `nv_acquire_mmap_lock` | nv-mmap.c | implemented (real mutex) | |
| `nv_release_mmap_lock` | nv-mmap.c | implemented | |
| `nv_get_all_mappings_revoked_locked` | nv-mmap.c | implemented (no user mappings exist: NV_TRUE) | |
| `nv_set_safe_to_mmap_locked` | nv-mmap.c | implemented | |
| `nv_revoke_gpu_mappings` | nv-mmap.c | implemented (nothing to revoke: NV_OK) | |
| `nv_i2c_add_adapter` | nv-i2c.c | refused | |
| `nv_i2c_del_adapter` | nv-i2c.c | refused | |
| `pci_enable_device`, `pci_request_regions` | shims/shim_pci.c | implemented: `-ENODEV` unless the identity gate accepted the device; no config write |  |
| `pci_disable_device`, `pci_set_master`, `pci_release_regions` | shims/shim_pci.c | stub: no-op (no command-register write; the command register is on the deny list) |  |
| `pci_read_config_{byte,word,dword}` | shims/shim_pci.c | implemented | kot_cfg_read (dword) |
| `pci_write_config_{byte,word,dword}` | shims/shim_pci.c | implemented (denied: `-EPERM`) | kot_cfg_write (policy, admission, dword RMW, ring log) |
| `pci_iomap`, `pci_iounmap` | shims/shim_pci.c | implemented | registry token for BAR0/BAR1 (sizes from the KPI); NULL for other BARs |
| `pci_register_driver` | shims/shim_pci.c | implemented: probes one `struct pci_dev` with the KPI BAR sizes and synthetic distinct bases; `-ENODEV` and no probe when the identity gate refuses |  |
| `ioremap`, `iounmap` | shims/shim_io.c | implemented | registry token / invalidation |
| `shim_readb/w/l/q` | shims/shim_io.c | implemented: token validated and bounds-checked; all-ones (0xff / 0xffff / ...) on any refusal or failed KPI read | cudaDriverKitKpiReadReg32 (dword) |
| `shim_writel/q` | shims/shim_io.c | implemented | write policy + shim_mutation_admit + cudaDriverKitKpiWriteReg32 + ring log |
| `shim_writeb/w` (byte/word RMW) | shims/shim_io.c | implemented: policy on the aligned dword; nothing written after a failed read | cudaDriverKitKpiReadReg32 + WriteReg32 (dword RMW) |
| `dma_alloc_coherent` | shims/shim_dma.c | implemented: identity-gated; zeroed (Linux contract) | shim_mutation_admit + cudaDriverKitKpiDmaAlloc (low32, 256-byte aligned) |
| `dma_free_coherent` | shims/shim_dma.c | implemented: quarantine unless quiesced; double free / unknown pointer / size mismatch reported | cudaDriverKitKpiDmaFree only once quiesced |
| `dma_map_single`, `dma_unmap_single` | shims/shim_dma.c | implemented: unmap never frees; refused on a quarantined buffer | cudaDriverKitKpiDmaSyncToDevice / SyncFromDevice |
| `dma_sync_single_{for_device,for_cpu}`, `dma_sync_single_range_*` | shims/shim_dma.c | implemented: refused on a quarantined buffer | cudaDriverKitKpiDmaSyncToDevice / SyncFromDevice |
| `dma_mapping_error` | shims/shim_dma.c | implemented: a quarantined buffer is an error |  |
| `kot_poll_interrupts` | (this layer) | implemented: runs the recorded ISR exclusive against every spinlock holder (gives up after 10 ms, returns -3), then the bottom half |  |
| `kot_irq_poll_thread_start/stop` | (this layer) | implemented, not started by default: calls `kot_poll_interrupts` at a fixed period |  |
| worker thread (`os_queue_work_item`, `kot_work_post`) | (this layer) | implemented: one FIFO thread for RM items and the layer's timer expiries; `kot_work_queue_stop` refuses new items, drains, ends it (teardown) |  |
| timer thread (`kot_timer_macos.c`) | (this layer) | implemented: one thread, posts expiries to the worker; `kot_timer_shutdown` ends it |  |
| `kot_cpu_read_midr` (MIDR_EL1 for platform/cpu.c) | (EL1 register) | stub: Apple implementer 0x61, part 0 | |
| `kot_cpu_read_ccsidr` (CCSIDR_EL1) | (EL1 register) | implemented (encoded from sysctl cache sizes) | |
| `kot_cpu_write_csselr` (CSSELR_EL1) | (EL1 register) | implemented (selects the level kot_cpu_read_ccsidr reports) | |

Implementation files: `os_interface_macos.c`, `os_pci_macos.c`, `os_mlock_macos.c`, `os_registry_macos.c`, `nv_macos.c`, `os_cpu_macos.c`, `kot_timer_macos.c`, and the plain-C `kot_lifecycle.c` (quarantine flag, `kot_fatal`), `kot_device.c` (identity, BARs, config choke point), `kot_write_policy.c` (policy table, allowlist, ring log); headers `macos_os_priv.h`, `kot_lifecycle.h`, `kot_device.h`, `kot_tokens.h`, `kot_write_policy.h`. Closure-only override headers in `override/`: `nv-priv.h` (sends `NV_PRIV_REG_RD/WR{08,16,32}` to `shim_readb/w/l` and `shim_writeb/w/l`, which call `cudaDriverKitKpiReadReg32`/`WriteReg32`) and `kot_cpu_arm.h` (sends the AArch64 EL1 system-register reads in `platform/cpu.c` to `os_cpu_macos.c`).

The `libspdm_*` crypto glue of `kernel-open/nvidia/libspdm_*.c` (confidential compute) is not implemented here. Those symbols are trap stubs in the closure link (`tools/closure/gen_trap_stubs.py`): reaching one aborts and names the symbol.
