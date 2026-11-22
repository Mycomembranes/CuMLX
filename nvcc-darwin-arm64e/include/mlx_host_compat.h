// mlx_host_compat.h — Darwin host-compile compatibility for MLX's CUDA backend.
//
// Force-included (via `-include`) by the nvcc-darwin shim's compile_host() for
// every HOST .cpp/.cc TU. It papers over POSIX gaps that CCCL / libcu++ assume
// but macOS does not provide, WITHOUT patching the fetched cccl-src tree. (The
// device-compile counterpart is mlx_device_compat.h.)
//
//   * sem_timedwait — CCCL's <cuda/std/__thread/threading_support_pthread.h>
//     calls POSIX sem_timedwait(), which macOS's <semaphore.h> neither declares
//     nor implements (macOS unnamed POSIX semaphores are stubs; only named
//     sem_open semaphores work, and even those lack the timed wait). We supply a
//     portable static-inline emulation that polls sem_trywait() until the
//     absolute deadline. static => internal linkage (one copy per TU, no ODR or
//     duplicate-symbol problem) and it is only defined on Apple, so a real Linux
//     + nvcc build is unaffected. This unblocks COMPILATION of the CCCL-using
//     host TUs (device.cpp / worker.cpp / …); the semaphore is exercised only at
//     runtime, and real bit-exact execution is sp3-canonical regardless.

#pragma once

#if defined(__APPLE__)
#include <semaphore.h>
#include <time.h>
#include <errno.h>

// macOS lacks sem_timedwait(2); emulate it by polling sem_trywait() with a small
// backoff until the caller's CLOCK_REALTIME absolute deadline elapses. Returns 0
// on acquire, -1 with errno=ETIMEDOUT on deadline, -1 on a real sem error.
static inline int sem_timedwait(sem_t* __sem, const struct timespec* __abstime) {
  for (;;) {
    if (sem_trywait(__sem) == 0) {
      return 0;
    }
    if (errno != EAGAIN) {
      return -1;  // a genuine error from sem_trywait (not "would block")
    }
    if (__abstime != 0) {
      struct timespec __now;
      clock_gettime(CLOCK_REALTIME, &__now);
      if (__now.tv_sec > __abstime->tv_sec ||
          (__now.tv_sec == __abstime->tv_sec &&
           __now.tv_nsec >= __abstime->tv_nsec)) {
        errno = ETIMEDOUT;
        return -1;
      }
    }
    struct timespec __slice;
    __slice.tv_sec = 0;
    __slice.tv_nsec = 200000;  // 0.2 ms poll interval
    nanosleep(&__slice, 0);
  }
}
#endif  // __APPLE__
