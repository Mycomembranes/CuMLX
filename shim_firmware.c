/*
 * shims/shim_firmware.c -- request_firmware() for the nvidia-ko translator.
 *
 * KOT_SAFE_SHIMS variant (RM closure link, shims/macos_os, host tests):
 *   - NVIDIA_KO_FIRMWARE_ROOT is required and must be an absolute path (no
 *     "firmware" default, no relative root);
 *   - no alias fallback: a request names one file, "nvidia/<version>/<file>"
 *     (as nv.c builds it; <version> must equal the OGK version this layer was
 *     built for, KOT_OGK_VERSION_STRING) or a bare "<file>", and is served only
 *     from <root>/<file>;
 *   - only pinned files are served: for 570.190 exactly the four files and
 *     SHA256 pins of scripts/egpu_reference_firmware_570190.py and
 *     crates/nvgpu-daemon-mac/src/bin/egpu_inproc_probe.rs (const PINS). The
 *     file must be a regular file (not a symlink) of the pinned size, and the
 *     SHA256 is computed over the bytes handed to RM. Any other file, version
 *     or mismatch is refused (-2, stderr line).
 * Without KOT_SAFE_SHIMS (default 570.190 / 595.71.05 build): the legacy
 * Linux-name alias resolver below the #else, kept byte-identical; NOT fit to
 * be put in front of a card.
 */
#if defined(KOT_SAFE_SHIMS)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

#include <CommonCrypto/CommonDigest.h>
#include <linux/firmware.h>
#include <shim_mutation_admission.h>

#ifndef KOT_OGK_VERSION_STRING
#define KOT_OGK_VERSION_STRING ""
#endif

struct shim_fw_pin {
    const char *version, *file;
    uint64_t size;
    const char *sha256;
    enum shim_firmware_role role;
};
static const struct shim_fw_pin k_pins[] = {
    { "570.190", "gsp_ga10x.bin", 64108272ull,
      "eb92d12eef04eafac12e31f50ebef097aa236cb389ac651496f8f12e38d1aadd", SHIM_FW_ROLE_GSP_RM },
    { "570.190", "fmc-570.190.bin", 199820ull,
      "e0950cdd16fe7045adbce11543f5b6d13146fca89e78158a9ab07d28dcbe576a", SHIM_FW_ROLE_FMC },
    { "570.190", "bootloader-570.190.bin", 200812ull,
      "e5e505fb453bcd618a4966464aa601e97a1ced2b1d450cab5bcb69e4043dc7f2", SHIM_FW_ROLE_BOOTLOADER },
    { "570.190", "booter_load-570.190.bin", 57720ull,
      "8b293e19b637c5e22c87a2428d1c71bb13e0904e8a88ac6b3c6c1f2679c6e37a", SHIM_FW_ROLE_BOOTER_LOAD },
};

static struct shim_firmware_resolution g_last_resolution;

static void shim_fw_refuse(const char *name, const char *why)
{
    fprintf(stderr, "kot-firmware: %s refused: %s\n", name ? name : "(null)", why);
}

/* The file part of a request, or NULL if the request is not one pinned file. */
static const struct shim_fw_pin *shim_fw_pin_for(const char *name, const char **why)
{
    const char *file = name;
    const char *ver = KOT_OGK_VERSION_STRING;
    if (strncmp(name, "nvidia/", 7) == 0) {
        size_t vl = strlen(ver);
        if (vl == 0 || strncmp(name + 7, ver, vl) != 0 || name[7 + vl] != '/') {
            *why = "the requested version directory is not the version this layer was built for";
            return NULL;
        }
        file = name + 7 + vl + 1;
    }
    if (file[0] == 0 || strchr(file, '/') != NULL || strchr(file, '\\') != NULL || strcmp(file, "..") == 0 ||
        strcmp(file, ".") == 0) {
        *why = "not a single file name";
        return NULL;
    }
    for (size_t i = 0; i < sizeof(k_pins) / sizeof(k_pins[0]); i++)
        if (strcmp(k_pins[i].version, ver) == 0 && strcmp(k_pins[i].file, file) == 0)
            return &k_pins[i];
    *why = "not a pinned firmware file for this version";
    return NULL;
}

int request_firmware(const struct firmware **fw, const char *name, struct device *device)
{
    (void)device;
    if (fw == 0 || name == 0) {
        return -1;
    }
    *fw = 0;
    memset(&g_last_resolution, 0, sizeof(g_last_resolution));
    snprintf(g_last_resolution.requested, sizeof(g_last_resolution.requested), "%s", name);
    snprintf(g_last_resolution.version, sizeof(g_last_resolution.version), "%s", KOT_OGK_VERSION_STRING);
    const char *root = getenv("NVIDIA_KO_FIRMWARE_ROOT");
    if (root == 0 || root[0] != '/') {
        shim_fw_refuse(name, "NVIDIA_KO_FIRMWARE_ROOT must be set to an absolute path");
        return -2;
    }
    const char *why = "";
    const struct shim_fw_pin *pin = shim_fw_pin_for(name, &why);
    if (pin == NULL) {
        shim_fw_refuse(name, why);
        return -2;
    }
    g_last_resolution.role = (u32)pin->role;
    if (shim_mutation_admit(CUDK_MUTATION_OP_FIRMWARE_STAGE) != 0) {
        return -2000;
    }
    char path[1024];
    if ((size_t)snprintf(path, sizeof(path), "%s/%s", root, pin->file) >= sizeof(path)) {
        shim_fw_refuse(name, "path too long");
        return -2;
    }
    struct stat st;
    if (lstat(path, &st) != 0 || !S_ISREG(st.st_mode)) {
        shim_fw_refuse(name, "the pinned file is missing or not a regular file (symlinks are refused)");
        return -2;
    }
    if ((uint64_t)st.st_size != pin->size) {
        shim_fw_refuse(name, "size differs from the pin");
        return -2;
    }
    FILE *f = fopen(path, "rb");
    unsigned char *data = (unsigned char *)malloc((size_t)pin->size);
    struct firmware *out = (struct firmware *)calloc(1, sizeof(*out));
    size_t got = (f && data) ? fread(data, 1, (size_t)pin->size, f) : 0;
    int extra = f ? fgetc(f) : EOF;
    if (f)
        fclose(f);
    if (data == 0 || out == 0 || got != pin->size || extra != EOF) {
        free(data);
        free(out);
        shim_fw_refuse(name, "read failed or the file changed size");
        return -2;
    }
    unsigned char md[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    for (uint64_t off = 0; off < pin->size; off += 1u << 20) {
        uint64_t n = pin->size - off < (1u << 20) ? pin->size - off : (1u << 20);
        CC_SHA256_Update(&ctx, data + off, (CC_LONG)n);
    }
    CC_SHA256_Final(md, &ctx);
    char hex[2 * CC_SHA256_DIGEST_LENGTH + 1];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++)
        snprintf(hex + 2 * i, 3, "%02x", md[i]);
    if (strcmp(hex, pin->sha256) != 0) {
        free(data);
        free(out);
        shim_fw_refuse(name, "SHA256 differs from the pin");
        return -2;
    }
    out->size = (size_t)pin->size;
    out->data = data;
    *fw = out;
    g_last_resolution.size = pin->size;
    snprintf(g_last_resolution.resolved_path, sizeof(g_last_resolution.resolved_path), "%s", path);
    fprintf(stderr, "kot-firmware: %s -> %s (%llu bytes, sha256 %s pinned)\n", name, path,
            (unsigned long long)pin->size, pin->sha256);
    return 0;
}

int firmware_request_nowarn(const struct firmware **fw, const char *name, struct device *device)
{
    return request_firmware(fw, name, device);
}

void release_firmware(const struct firmware *fw)
{
    if (fw != 0) {
        free((void *)fw->data);
        free((void *)fw);
    }
}

int request_firmware_nowait(struct module *module,
                            bool uevent,
                            const char *name,
                            struct device *device,
                            gfp_t gfp,
                            void *context,
                            void (*cont)(const struct firmware *fw, void *context))
{
    (void)module;
    (void)uevent;
    (void)gfp;
    const struct firmware *fw = 0;
    int rc = request_firmware(&fw, name, device);
    if (cont != 0) {
        cont(fw, context);
    }
    return rc;
}

int shim_firmwareLastResolution(struct shim_firmware_resolution *out, uint64_t out_size)
{
    if (out == 0 || out_size < sizeof(*out)) {
        return -1;
    }
    *out = g_last_resolution;
    return 0;
}

#else /* !KOT_SAFE_SHIMS: legacy default-build variant, unchanged */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <linux/firmware.h>
#include <shim_mutation_admission.h>

static struct shim_firmware_resolution g_last_resolution;

static void shim_copy_cstr(char *dst, size_t dst_len, const char *src)
{
    if (dst_len == 0) {
        return;
    }
    if (src == 0) {
        dst[0] = 0;
        return;
    }
    snprintf(dst, dst_len, "%s", src);
}

static const char *shim_firmware_root(void)
{
    const char *root = getenv("NVIDIA_KO_FIRMWARE_ROOT");
    if (root == 0 || root[0] == 0) {
        root = "firmware";
    }
    return root;
}

static char *shim_firmware_path(const char *root, const char *name)
{
    size_t n = strlen(root) + 1 + strlen(name) + 1;
    char *path = (char *)malloc(n);
    if (path != 0) {
        snprintf(path, n, "%s/%s", root, name);
    }
    return path;
}

static int shim_ascii_lower(int ch)
{
    if (ch >= 'A' && ch <= 'Z') {
        return ch + ('a' - 'A');
    }
    return ch;
}

static int shim_contains_ci(const char *haystack, const char *needle)
{
    size_t needle_len = strlen(needle);
    if (needle_len == 0) {
        return 1;
    }
    for (const char *h = haystack; *h != 0; h++) {
        size_t i = 0;
        while (i < needle_len &&
               h[i] != 0 &&
               shim_ascii_lower((unsigned char)h[i]) ==
                   shim_ascii_lower((unsigned char)needle[i])) {
            i++;
        }
        if (i == needle_len) {
            return 1;
        }
    }
    return 0;
}

static int shim_starts_ci(const char *value, const char *prefix)
{
    for (size_t i = 0; prefix[i] != 0; i++) {
        if (value[i] == 0 ||
            shim_ascii_lower((unsigned char)value[i]) !=
                shim_ascii_lower((unsigned char)prefix[i])) {
            return 0;
        }
    }
    return 1;
}

static const char *shim_basename(const char *name)
{
    const char *base = name;
    for (const char *p = name; *p != 0; p++) {
        if (*p == '/' || *p == '\\') {
            base = p + 1;
        }
    }
    return base;
}

static char *shim_extract_version(const char *name)
{
    for (const char *p = name; *p != 0; p++) {
        if (*p < '0' || *p > '9') {
            continue;
        }

        const char *start = p;
        while (*p >= '0' && *p <= '9') {
            p++;
        }
        if (*p != '.') {
            continue;
        }
        p++;
        const char *after_dot = p;
        while (*p >= '0' && *p <= '9') {
            p++;
        }
        if (p == after_dot) {
            continue;
        }

        size_t len = (size_t)(p - start);
        char *version = (char *)malloc(len + 1);
        if (version == 0) {
            return 0;
        }
        memcpy(version, start, len);
        version[len] = 0;
        return version;
    }
    return 0;
}

static void shim_firmware_trace_begin(const char *name,
                                      enum shim_firmware_role role,
                                      const char *version)
{
    memset(&g_last_resolution, 0, sizeof(g_last_resolution));
    g_last_resolution.role = (uint32_t)role;
    shim_copy_cstr(g_last_resolution.requested,
                   sizeof(g_last_resolution.requested),
                   name);
    shim_copy_cstr(g_last_resolution.version,
                   sizeof(g_last_resolution.version),
                   version);
}

static void shim_firmware_trace_success(const char *resolved_path,
                                        const char *resolved_name,
                                        const struct firmware *fw)
{
    shim_copy_cstr(g_last_resolution.resolved_path,
                   sizeof(g_last_resolution.resolved_path),
                   resolved_path);
    g_last_resolution.size = fw == 0 ? 0 : (uint64_t)fw->size;
    if (resolved_name != 0 && g_last_resolution.requested[0] != 0) {
        g_last_resolution.alias_used =
            strcmp(g_last_resolution.requested, resolved_name) == 0 ? 0U : 1U;
        if (g_last_resolution.alias_used != 0) {
            shim_copy_cstr(g_last_resolution.alias,
                           sizeof(g_last_resolution.alias),
                           resolved_name);
        }
    }
}

static enum shim_firmware_role shim_firmware_role_for_name(const char *name)
{
    if (shim_contains_ci(name, "fwsec")) {
        return SHIM_FW_ROLE_FWSEC;
    }
    if (shim_contains_ci(name, "booter_unload") ||
        shim_contains_ci(name, "booter-unload") ||
        shim_contains_ci(name, "booterunload")) {
        return SHIM_FW_ROLE_BOOTER_UNLOAD;
    }
    if (shim_contains_ci(name, "booter_load") ||
        shim_contains_ci(name, "booter-load") ||
        shim_contains_ci(name, "booterload") ||
        shim_contains_ci(name, "booter")) {
        return SHIM_FW_ROLE_BOOTER_LOAD;
    }
    if (shim_contains_ci(name, "bootloader") ||
        shim_contains_ci(name, "gsprmboot") ||
        shim_contains_ci(name, "gsp_rm_boot") ||
        shim_contains_ci(name, "gsp-rm-boot")) {
        return SHIM_FW_ROLE_BOOTLOADER;
    }
    if (shim_contains_ci(name, "fmc") || shim_contains_ci(name, "gfw")) {
        return SHIM_FW_ROLE_FMC;
    }
    if (shim_contains_ci(name, "bindata")) {
        return SHIM_FW_ROLE_BINDATA;
    }
    const char *base = shim_basename(name);
    if (!shim_contains_ci(name, "gsp_log") &&
        (shim_contains_ci(name, "gsp_rm") ||
         shim_contains_ci(name, "gsp-rm") ||
         shim_starts_ci(base, "gsp_ga10x") ||
         shim_starts_ci(base, "gsp_tu10x") ||
         shim_starts_ci(base, "gsp-"))) {
        return SHIM_FW_ROLE_GSP_RM;
    }
    return SHIM_FW_ROLE_NONE;
}

static int shim_read_firmware_file(FILE *f, const struct firmware **fw)
{
    if (fseek(f, 0, SEEK_END) != 0) {
        return -1;
    }
    long len = ftell(f);
    if (len < 0 || fseek(f, 0, SEEK_SET) != 0) {
        return -1;
    }
    struct firmware *out = (struct firmware *)calloc(1, sizeof(*out));
    size_t alloc_len = len == 0 ? 1 : (size_t)len;
    unsigned char *data = (unsigned char *)malloc(alloc_len);
    if (out == 0 || data == 0) {
        free(out);
        free(data);
        return -1;
    }
    if (fread(data, 1, (size_t)len, f) != (size_t)len) {
        free(out);
        free(data);
        return -1;
    }
    out->size = (size_t)len;
    out->data = data;
    *fw = out;
    return 0;
}

static int shim_request_firmware_path(const char *root, const char *name, const struct firmware **fw)
{
    char *path = shim_firmware_path(root, name);
    if (path == 0) {
        return -1;
    }
    FILE *f = fopen(path, "rb");
    if (f == 0) {
        free(path);
        return -1;
    }
    int rc = shim_read_firmware_file(f, fw);
    fclose(f);
    if (rc == 0) {
        shim_firmware_trace_success(path, name, *fw);
    }
    free(path);
    return rc;
}

static int shim_try_versioned(const char *root,
                              const char *prefix,
                              const char *version,
                              const char *suffix,
                              const struct firmware **fw)
{
    if (version == 0 || version[0] == 0) {
        return -1;
    }
    size_t n = strlen(prefix) + strlen(version) + strlen(suffix) + 1;
    char *name = (char *)malloc(n);
    if (name == 0) {
        return -1;
    }
    snprintf(name, n, "%s%s%s", prefix, version, suffix);
    int rc = shim_request_firmware_path(root, name, fw);
    free(name);
    return rc;
}

static int shim_try_fwsec_aliases(const char *root,
                                  const char *name,
                                  const char *version,
                                  const struct firmware **fw)
{
    if (shim_contains_ci(name, "desc")) {
        if (shim_request_firmware_path(root, "fwsec_desc.bin", fw) == 0 ||
            shim_try_versioned(root, "fwsec-desc-", version, ".bin", fw) == 0) {
            return 0;
        }
    }
    if (shim_contains_ci(name, "sig")) {
        if (shim_request_firmware_path(root, "fwsec_sigs.bin", fw) == 0 ||
            shim_request_firmware_path(root, "fwsec_sig.bin", fw) == 0 ||
            shim_try_versioned(root, "fwsec-sigs-", version, ".bin", fw) == 0 ||
            shim_try_versioned(root, "fwsec-sig-", version, ".bin", fw) == 0) {
            return 0;
        }
    }
    if (shim_contains_ci(name, "ucode") || shim_contains_ci(name, "image")) {
        if (shim_request_firmware_path(root, "fwsec_ucode.bin", fw) == 0 ||
            shim_try_versioned(root, "fwsec-ucode-", version, ".bin", fw) == 0) {
            return 0;
        }
    }
    if (shim_try_versioned(root, "fwsec-", version, ".bin", fw) == 0 ||
        shim_request_firmware_path(root, "fwsec.bin", fw) == 0) {
        return 0;
    }
    return -1;
}

static int shim_try_firmware_aliases(const char *root, const char *name, const struct firmware **fw)
{
    enum shim_firmware_role role = shim_firmware_role_for_name(name);
    if (role == SHIM_FW_ROLE_NONE) {
        return -1;
    }

    const char *base = shim_basename(name);
    char *version = shim_extract_version(name);

    if (base[0] != 0 && shim_request_firmware_path(root, base, fw) == 0) {
        free(version);
        return 0;
    }

    int rc = -1;
    switch (role) {
        case SHIM_FW_ROLE_GSP_RM:
            if (shim_request_firmware_path(root, "gsp_ga10x.bin", fw) == 0 ||
                shim_try_versioned(root, "gsp-", version, ".bin", fw) == 0 ||
                shim_try_versioned(root, "gsp_rm-", version, ".bin", fw) == 0 ||
                shim_request_firmware_path(root, "gsp_rm.bin", fw) == 0 ||
                shim_request_firmware_path(root, "gsp.bin", fw) == 0) {
                rc = 0;
            }
            break;
        case SHIM_FW_ROLE_FMC:
            if (shim_try_versioned(root, "fmc-", version, ".bin", fw) == 0 ||
                shim_request_firmware_path(root, "gb202_fmc_202_ucode_image_data.bin", fw) == 0 ||
                shim_request_firmware_path(root, "fmc.bin", fw) == 0 ||
                shim_request_firmware_path(root, "gsp_fmc.bin", fw) == 0) {
                rc = 0;
            }
            break;
        case SHIM_FW_ROLE_BOOTER_LOAD:
            if (shim_try_versioned(root, "booter_load-", version, ".bin", fw) == 0 ||
                shim_request_firmware_path(root, "booter_load.bin", fw) == 0 ||
                shim_try_versioned(root, "booter-", version, ".bin", fw) == 0 ||
                shim_request_firmware_path(root, "booter.bin", fw) == 0) {
                rc = 0;
            }
            break;
        case SHIM_FW_ROLE_BOOTER_UNLOAD:
            if (shim_try_versioned(root, "booter_unload-", version, ".bin", fw) == 0 ||
                shim_request_firmware_path(root, "booter_unload.bin", fw) == 0) {
                rc = 0;
            }
            break;
        case SHIM_FW_ROLE_BOOTLOADER:
            if (shim_try_versioned(root, "bootloader-", version, ".bin", fw) == 0 ||
                shim_request_firmware_path(root, "bootloader.bin", fw) == 0 ||
                shim_try_versioned(root, "gsp_bootloader-", version, ".bin", fw) == 0) {
                rc = 0;
            }
            break;
        case SHIM_FW_ROLE_FWSEC:
            rc = shim_try_fwsec_aliases(root, name, version, fw);
            break;
        case SHIM_FW_ROLE_BINDATA:
            if (shim_request_firmware_path(root, "bindata_image.bin", fw) == 0 ||
                shim_try_versioned(root, "bindata-", version, ".bin", fw) == 0 ||
                shim_request_firmware_path(root, "bindata.bin", fw) == 0) {
                rc = 0;
            }
            break;
        case SHIM_FW_ROLE_NONE:
            break;
    }

    free(version);
    return rc;
}

int request_firmware(const struct firmware **fw, const char *name, struct device *device)
{
    (void)device;
    if (fw == 0 || name == 0) {
        return -1;
    }
    *fw = 0;
    uint32_t role = shim_firmware_role_for_name(name);
    char *version = shim_extract_version(name);
    shim_firmware_trace_begin(name, role, version);
    free(version);
    if (role != SHIM_FW_ROLE_NONE &&
        shim_mutation_admit(CUDK_MUTATION_OP_FIRMWARE_STAGE) != 0) {
        return -2000;
    }
    const char *root = shim_firmware_root();
    if (shim_try_firmware_aliases(root, name, fw) == 0) {
        return 0;
    }
    return shim_request_firmware_path(root, name, fw);
}

int firmware_request_nowarn(const struct firmware **fw, const char *name, struct device *device)
{
    return request_firmware(fw, name, device);
}

void release_firmware(const struct firmware *fw)
{
    if (fw != 0) {
        free((void *)fw->data);
        free((void *)fw);
    }
}

int request_firmware_nowait(struct module *module,
                            bool uevent,
                            const char *name,
                            struct device *device,
                            gfp_t gfp,
                            void *context,
                            void (*cont)(const struct firmware *fw, void *context))
{
    (void)module;
    (void)uevent;
    (void)gfp;
    const struct firmware *fw = 0;
    int rc = request_firmware(&fw, name, device);
    if (cont != 0) {
        cont(fw, context);
    }
    return rc;
}

int shim_firmwareLastResolution(struct shim_firmware_resolution *out, uint64_t out_size)
{
    if (out == 0 || out_size < sizeof(*out)) {
        return -1;
    }
    *out = g_last_resolution;
    return 0;
}
#endif /* KOT_SAFE_SHIMS */
