/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * fwd_wire.h - the small msgpack subset and framed AF_UNIX transport used by
 * the stand-in libcuda forwarder (cuda-oxide.driver.v1).
 *
 * Self-contained C with no third-party dependency, so the same source builds
 * for x86_64 (Rosetta clients) and arm64. Everything here is file-local.
 */
#ifndef CUDA_ADAPTER_FWD_WIRE_H
#define CUDA_ADAPTER_FWD_WIRE_H

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

/* ---- writer ------------------------------------------------------------ */

typedef struct {
    uint8_t *data;
    size_t len;
    size_t cap;
    int failed; /* allocation failure: the request is abandoned */
} mp_buf;

static void mp_reserve(mp_buf *b, size_t extra) {
    if (b->failed || b->len + extra <= b->cap) return;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->len + extra) cap *= 2;
    uint8_t *grown = (uint8_t *)realloc(b->data, cap);
    if (!grown) {
        b->failed = 1;
        return;
    }
    b->data = grown;
    b->cap = cap;
}

static void mp_raw(mp_buf *b, const void *src, size_t n) {
    mp_reserve(b, n);
    if (b->failed) return;
    memcpy(b->data + b->len, src, n);
    b->len += n;
}

static void mp_byte(mp_buf *b, uint8_t v) { mp_raw(b, &v, 1); }

static void mp_be(mp_buf *b, uint64_t v, int bytes) {
    uint8_t tmp[8];
    for (int i = 0; i < bytes; i++) tmp[i] = (uint8_t)(v >> (8 * (bytes - 1 - i)));
    mp_raw(b, tmp, (size_t)bytes);
}

static void mp_uint(mp_buf *b, uint64_t v) {
    if (v < 0x80) {
        mp_byte(b, (uint8_t)v);
    } else if (v <= 0xff) {
        mp_byte(b, 0xcc);
        mp_be(b, v, 1);
    } else if (v <= 0xffff) {
        mp_byte(b, 0xcd);
        mp_be(b, v, 2);
    } else if (v <= 0xffffffffull) {
        mp_byte(b, 0xce);
        mp_be(b, v, 4);
    } else {
        mp_byte(b, 0xcf);
        mp_be(b, v, 8);
    }
}

static void mp_int(mp_buf *b, int64_t v) {
    if (v >= 0) {
        mp_uint(b, (uint64_t)v);
    } else {
        mp_byte(b, 0xd3);
        mp_be(b, (uint64_t)v, 8);
    }
}

static void mp_str(mp_buf *b, const char *s) {
    size_t n = strlen(s);
    if (n < 32) {
        mp_byte(b, (uint8_t)(0xa0 | n));
    } else if (n <= 0xff) {
        mp_byte(b, 0xd9);
        mp_be(b, n, 1);
    } else if (n <= 0xffff) {
        mp_byte(b, 0xda);
        mp_be(b, n, 2);
    } else {
        mp_byte(b, 0xdb);
        mp_be(b, n, 4);
    }
    mp_raw(b, s, n);
}

static void mp_bin(mp_buf *b, const void *src, size_t n) {
    if (n <= 0xff) {
        mp_byte(b, 0xc4);
        mp_be(b, n, 1);
    } else if (n <= 0xffff) {
        mp_byte(b, 0xc5);
        mp_be(b, n, 2);
    } else {
        mp_byte(b, 0xc6);
        mp_be(b, n, 4);
    }
    mp_raw(b, src, n);
}

static void mp_map(mp_buf *b, uint32_t pairs) {
    if (pairs < 16) {
        mp_byte(b, (uint8_t)(0x80 | pairs));
    } else {
        mp_byte(b, 0xde);
        mp_be(b, pairs, 2);
    }
}

static void mp_array(mp_buf *b, uint32_t items) {
    if (items < 16) {
        mp_byte(b, (uint8_t)(0x90 | items));
    } else {
        mp_byte(b, 0xdc);
        mp_be(b, items, 2);
    }
}

/* ---- reader ------------------------------------------------------------ */

typedef struct {
    const uint8_t *p;
    const uint8_t *end;
    int bad; /* truncated or unexpected type */
} mp_cur;

static uint64_t mp_take_be(mp_cur *c, int bytes) {
    if (c->bad || c->end - c->p < bytes) {
        c->bad = 1;
        return 0;
    }
    uint64_t v = 0;
    for (int i = 0; i < bytes; i++) v = (v << 8) | c->p[i];
    c->p += bytes;
    return v;
}

static uint8_t mp_peek(mp_cur *c) {
    if (c->bad || c->p >= c->end) {
        c->bad = 1;
        return 0xc1; /* never-used msgpack byte */
    }
    return *c->p;
}

/* Read a map or array header; returns the element count. */
static uint32_t mp_read_container(mp_cur *c, int want_map) {
    uint8_t t = mp_peek(c);
    if (c->bad) return 0;
    c->p++;
    if (want_map) {
        if ((t & 0xf0) == 0x80) return t & 0x0f;
        if (t == 0xde) return (uint32_t)mp_take_be(c, 2);
        if (t == 0xdf) return (uint32_t)mp_take_be(c, 4);
    } else {
        if ((t & 0xf0) == 0x90) return t & 0x0f;
        if (t == 0xdc) return (uint32_t)mp_take_be(c, 2);
        if (t == 0xdd) return (uint32_t)mp_take_be(c, 4);
    }
    c->bad = 1;
    return 0;
}

/* Read a str or bin; returns a pointer into the reply and its length. */
static const uint8_t *mp_read_bytes(mp_cur *c, size_t *len, int want_str) {
    uint8_t t = mp_peek(c);
    *len = 0;
    if (c->bad) return NULL;
    c->p++;
    size_t n;
    if (want_str && (t & 0xe0) == 0xa0) n = t & 0x1f;
    else if (want_str && t == 0xd9) n = (size_t)mp_take_be(c, 1);
    else if (want_str && t == 0xda) n = (size_t)mp_take_be(c, 2);
    else if (want_str && t == 0xdb) n = (size_t)mp_take_be(c, 4);
    else if (!want_str && t == 0xc4) n = (size_t)mp_take_be(c, 1);
    else if (!want_str && t == 0xc5) n = (size_t)mp_take_be(c, 2);
    else if (!want_str && t == 0xc6) n = (size_t)mp_take_be(c, 4);
    else {
        c->bad = 1;
        return NULL;
    }
    if (c->bad || (size_t)(c->end - c->p) < n) {
        c->bad = 1;
        return NULL;
    }
    const uint8_t *start = c->p;
    c->p += n;
    *len = n;
    return start;
}

static int64_t mp_read_int(mp_cur *c) {
    uint8_t t = mp_peek(c);
    if (c->bad) return 0;
    c->p++;
    if (t < 0x80) return t;
    if (t >= 0xe0) return (int8_t)t;
    switch (t) {
    case 0xcc: return (int64_t)mp_take_be(c, 1);
    case 0xcd: return (int64_t)mp_take_be(c, 2);
    case 0xce: return (int64_t)mp_take_be(c, 4);
    case 0xcf: return (int64_t)mp_take_be(c, 8);
    case 0xd0: return (int8_t)mp_take_be(c, 1);
    case 0xd1: return (int16_t)mp_take_be(c, 2);
    case 0xd2: return (int32_t)mp_take_be(c, 4);
    case 0xd3: return (int64_t)mp_take_be(c, 8);
    default:
        c->bad = 1;
        return 0;
    }
}

/* Skip one value of any supported type. */
static void mp_skip(mp_cur *c) {
    uint8_t t = mp_peek(c);
    if (c->bad) return;
    size_t len;
    if (t == 0xc0 || t == 0xc2 || t == 0xc3) {
        c->p++;
    } else if ((t & 0xf0) == 0x80 || t == 0xde || t == 0xdf) {
        uint32_t n = mp_read_container(c, 1);
        for (uint32_t i = 0; i < n * 2 && !c->bad; i++) mp_skip(c);
    } else if ((t & 0xf0) == 0x90 || t == 0xdc || t == 0xdd) {
        uint32_t n = mp_read_container(c, 0);
        for (uint32_t i = 0; i < n && !c->bad; i++) mp_skip(c);
    } else if ((t & 0xe0) == 0xa0 || t == 0xd9 || t == 0xda || t == 0xdb) {
        (void)mp_read_bytes(c, &len, 1);
    } else if (t == 0xc4 || t == 0xc5 || t == 0xc6) {
        (void)mp_read_bytes(c, &len, 0);
    } else if (t == 0xca) {
        c->p++;
        (void)mp_take_be(c, 4);
    } else if (t == 0xcb) {
        c->p++;
        (void)mp_take_be(c, 8);
    } else {
        (void)mp_read_int(c);
    }
}

/* Position `out` at the value stored under `key` in the map at `map`. */
static int mp_find(mp_cur map, const char *key, mp_cur *out) {
    uint32_t pairs = mp_read_container(&map, 1);
    size_t want = strlen(key);
    for (uint32_t i = 0; i < pairs && !map.bad; i++) {
        size_t len;
        const uint8_t *name = mp_read_bytes(&map, &len, 1);
        if (map.bad) return 0;
        if (len == want && memcmp(name, key, want) == 0) {
            *out = map;
            return 1;
        }
        mp_skip(&map);
    }
    return 0;
}

static int mp_is_nil(mp_cur c) { return !c.bad && c.p < c.end && *c.p == 0xc0; }

/* ---- framed transport -------------------------------------------------- */

#define FWD_MAX_FRAME (64u * 1024u * 1024u)

static int fwd_write_all(int fd, const void *src, size_t n) {
    const uint8_t *p = (const uint8_t *)src;
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        p += w;
        n -= (size_t)w;
    }
    return 0;
}

static int fwd_read_all(int fd, void *dst, size_t n) {
    uint8_t *p = (uint8_t *)dst;
    while (n) {
        ssize_t r = read(fd, p, n);
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (r == 0) return -1; /* peer closed */
        p += r;
        n -= (size_t)r;
    }
    return 0;
}

static int fwd_send_frame(int fd, const mp_buf *body) {
    uint8_t len[4] = {(uint8_t)(body->len >> 24), (uint8_t)(body->len >> 16), (uint8_t)(body->len >> 8),
                      (uint8_t)body->len};
    if (body->len > FWD_MAX_FRAME) return -1;
    if (fwd_write_all(fd, len, 4) != 0) return -1;
    return fwd_write_all(fd, body->data, body->len);
}

/* Returns a malloc'd body the caller frees, or NULL. */
static uint8_t *fwd_recv_frame(int fd, size_t *out_len) {
    uint8_t len[4];
    if (fwd_read_all(fd, len, 4) != 0) return NULL;
    size_t n = ((size_t)len[0] << 24) | ((size_t)len[1] << 16) | ((size_t)len[2] << 8) | len[3];
    if (n > FWD_MAX_FRAME) return NULL;
    uint8_t *body = (uint8_t *)malloc(n ? n : 1);
    if (!body) return NULL;
    if (fwd_read_all(fd, body, n) != 0) {
        free(body);
        return NULL;
    }
    *out_len = n;
    return body;
}

static int fwd_connect_unix(const char *path) {
    struct sockaddr_un addr;
    if (strlen(path) >= sizeof(addr.sun_path)) return -1;
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, path);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

/* Receive the marker byte and the descriptor the server attaches after a
 * host_alloc reply. Returns the descriptor, or -1. */
static int fwd_recv_fd(int sock) {
    uint8_t marker = 0;
    struct iovec iov = {&marker, 1};
    union {
        struct cmsghdr align;
        char bytes[CMSG_SPACE(sizeof(int))];
    } control;
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    memset(&control, 0, sizeof(control));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.bytes;
    msg.msg_controllen = sizeof(control.bytes);
    ssize_t n;
    do {
        n = recvmsg(sock, &msg, 0);
    } while (n < 0 && errno == EINTR);
    if (n != 1 || marker != 0xFD) return -1;
    for (struct cmsghdr *c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c)) {
        if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS && c->cmsg_len >= CMSG_LEN(sizeof(int))) {
            int fd;
            memcpy(&fd, CMSG_DATA(c), sizeof(fd));
            return fd;
        }
    }
    return -1;
}

#endif /* CUDA_ADAPTER_FWD_WIRE_H */
