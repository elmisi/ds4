#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int check_head_rms_rope_fusion(void) {
    const uint32_t n_tok = 2;
    const uint32_t n_head = 3;
    const uint32_t head_dim = 16;
    const uint32_t n_rot = 8;
    const uint64_t count = (uint64_t)n_tok * n_head * head_dim;
    float *input = (float *)malloc((size_t)count * sizeof(float));
    float *separate = (float *)malloc((size_t)count * sizeof(float));
    float *fused = (float *)malloc((size_t)count * sizeof(float));
    if (!input || !separate || !fused) return 1;

    for (uint64_t i = 0; i < count; i++) {
        input[i] = (float)((int)(i % 29) - 14) / 17.0f;
    }

    ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(count * sizeof(float));
    ds4_gpu_tensor *b = ds4_gpu_tensor_alloc(count * sizeof(float));
    int rc = 1;
    if (a && b &&
        ds4_gpu_tensor_write(a, 0, input, count * sizeof(float)) &&
        ds4_gpu_tensor_write(b, 0, input, count * sizeof(float)) &&
        ds4_gpu_head_rms_norm_tensor(a, n_tok, n_head, head_dim, 1e-6f) &&
        ds4_gpu_rope_tail_tensor(a, n_tok, n_head, head_dim, n_rot,
                                 37, 0, false,
                                 10000.0f, 1.0f, 0.0f, 1.0f,
                                 32.0f, 1.0f) &&
        ds4_gpu_head_rms_norm_rope_tail_tensor(b, n_tok, n_head, head_dim, n_rot,
                                               37, 0, false,
                                               10000.0f, 1.0f, 0.0f, 1.0f,
                                               32.0f, 1.0f, 1e-6f) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(a, 0, separate, count * sizeof(float)) &&
        ds4_gpu_tensor_read(b, 0, fused, count * sizeof(float))) {
        rc = 0;
        for (uint64_t i = 0; i < count; i++) {
            const float diff = fabsf(separate[i] - fused[i]);
            if (diff > 2e-4f) {
                fprintf(stderr,
                        "head rms+rope fusion mismatch index=%llu separate=%g fused=%g diff=%g\n",
                        (unsigned long long)i,
                        (double)separate[i],
                        (double)fused[i],
                        (double)diff);
                rc = 1;
                break;
            }
        }
    }

    ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(a);
    free(fused);
    free(separate);
    free(input);
    return rc;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    const int rc = check_head_rms_rope_fusion();
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda fused decode smoke: OK");
    return rc;
}
