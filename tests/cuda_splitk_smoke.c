#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static uint16_t f32_to_f16(float f) {
    union {
        float f;
        uint32_t u;
    } v = {f};
    uint32_t sign = (v.u >> 16) & 0x8000u;
    int exp = (int)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    union {
        uint32_t u;
        float f;
    } v = {bits};
    return v.f;
}

static uint64_t round_up_u64(uint64_t n, uint64_t align) {
    return (n + align - 1u) & ~(align - 1u);
}

static int run_case(int splitk, const uint16_t *weights, uint64_t weight_alloc,
                    const float *x_host, float *out_host,
                    uint32_t in_dim, uint32_t out_dim) {
    if (splitk) {
        setenv("DS4_CUDA_F16_SPLITK", "1", 1);
        unsetenv("DS4_CUDA_NO_F16_SPLITK");
    } else {
        unsetenv("DS4_CUDA_F16_SPLITK");
        setenv("DS4_CUDA_NO_F16_SPLITK", "1", 1);
    }
    unsetenv("DS4_CUDA_ORDERED_F16_MATMUL");
    unsetenv("DS4_CUDA_FORCE_ORDERED_F16_MATMUL");

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    int rc = 1;
    if (x && out &&
        ds4_gpu_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) &&
        ds4_gpu_set_model_map(weights, weight_alloc) &&
        ds4_gpu_matmul_f16_tensor(out, weights, weight_alloc, 0,
                                  in_dim, out_dim, x, 1) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(out, 0, out_host, (uint64_t)out_dim * sizeof(float))) {
        rc = 0;
    }
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    return rc;
}

int main(void) {
    const uint32_t in_dim = 4096;
    const uint32_t out_dim = 128;
    const uint64_t weight_bytes = (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = round_up_u64(weight_bytes, (uint64_t)getpagesize());

    void *weights_raw = NULL;
    if (posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) != 0) {
        return 1;
    }
    uint16_t *weights = (uint16_t *)weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((o * 17u + i * 19u + (o ^ i) * 3u) % 97u) - 48;
            weights[(uint64_t)o * in_dim + i] = f32_to_f16((float)v / 256.0f);
        }
    }

    float *x = (float *)malloc((size_t)in_dim * sizeof(float));
    float *split = (float *)malloc((size_t)out_dim * sizeof(float));
    float *nosplit = (float *)malloc((size_t)out_dim * sizeof(float));
    if (!x || !split || !nosplit) return 1;
    for (uint32_t i = 0; i < in_dim; i++) {
        const int v = (int)((i * 23u + (i >> 3) * 7u) % 89u) - 44;
        x[i] = (float)v / 192.0f;
    }

    if (run_case(1, weights, weight_alloc, x, split, in_dim, out_dim) != 0 ||
        run_case(0, weights, weight_alloc, x, nosplit, in_dim, out_dim) != 0) {
        free(nosplit);
        free(split);
        free(x);
        free(weights_raw);
        return 1;
    }

    float max_ref = 0.0f;
    float max_pair = 0.0f;
    for (uint32_t o = 0; o < out_dim; o++) {
        float ref = 0.0f;
        for (uint32_t i = 0; i < in_dim; i++) {
            ref += f16_to_f32(weights[(uint64_t)o * in_dim + i]) * x[i];
        }
        const float err = fabsf(split[o] - ref);
        const float diff = fabsf(split[o] - nosplit[o]);
        if (err > max_ref) max_ref = err;
        if (diff > max_pair) max_pair = diff;
        if (!isfinite(split[o]) || !isfinite(nosplit[o])) return 1;
    }

    printf("cuda split-k smoke: max_ref=%g max_vs_nosplit=%g\n",
           (double)max_ref, (double)max_pair);
    free(nosplit);
    free(split);
    free(x);
    free(weights_raw);
    return (max_ref < 0.01f && max_pair < 0.01f) ? 0 : 1;
}
