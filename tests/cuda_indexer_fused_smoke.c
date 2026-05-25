/* Smoke test for the CUDA fused decode indexer score+top-k path.
 *
 * It compares the new fused API against the existing score_one + topk pair on
 * deterministic synthetic tensors. No model file is needed.
 */

#include "ds4_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_HEAD 64u
#define HEAD_DIM 128u
#define N_COMP 8192u
#define TOP_K 512u

static uint32_t rng_state = 0x12345678u;

static float rndf(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    int32_t v = (int32_t)((rng_state >> 8) & 0xffffu) - 32768;
    return (float)v / 32768.0f;
}

static int write_tensor(ds4_gpu_tensor *t, const void *ptr, uint64_t bytes, const char *what) {
    if (ds4_gpu_tensor_write(t, 0, ptr, bytes)) return 1;
    fprintf(stderr, "%s write failed\n", what);
    return 0;
}

int main(void) {
    int rc = 1;
    float *q = NULL;
    float *weights = NULL;
    float *index_comp = NULL;
    uint32_t *ref = NULL;
    uint32_t *fused = NULL;
    ds4_gpu_tensor *t_q = NULL;
    ds4_gpu_tensor *t_weights = NULL;
    ds4_gpu_tensor *t_index_comp = NULL;
    ds4_gpu_tensor *t_scores = NULL;
    ds4_gpu_tensor *t_ref = NULL;
    ds4_gpu_tensor *t_fused = NULL;

    const uint64_t q_bytes = (uint64_t)N_HEAD * HEAD_DIM * sizeof(float);
    const uint64_t weights_bytes = (uint64_t)N_HEAD * sizeof(float);
    const uint64_t index_comp_bytes = (uint64_t)N_COMP * HEAD_DIM * sizeof(float);
    const uint64_t scores_bytes = (uint64_t)N_COMP * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)TOP_K * sizeof(uint32_t);

    q = (float *)malloc((size_t)q_bytes);
    weights = (float *)malloc((size_t)weights_bytes);
    index_comp = (float *)malloc((size_t)index_comp_bytes);
    ref = (uint32_t *)malloc((size_t)selected_bytes);
    fused = (uint32_t *)malloc((size_t)selected_bytes);
    if (!q || !weights || !index_comp || !ref || !fused) {
        fprintf(stderr, "host allocation failed\n");
        goto out;
    }

    for (uint64_t i = 0; i < (uint64_t)N_HEAD * HEAD_DIM; i++) q[i] = rndf();
    for (uint32_t i = 0; i < N_HEAD; i++) weights[i] = 0.2f + fabsf(rndf());
    for (uint64_t i = 0; i < (uint64_t)N_COMP * HEAD_DIM; i++) index_comp[i] = rndf();

    if (!ds4_gpu_init()) {
        fprintf(stderr, "ds4_gpu_init failed\n");
        goto out;
    }

    t_q = ds4_gpu_tensor_alloc(q_bytes);
    t_weights = ds4_gpu_tensor_alloc(weights_bytes);
    t_index_comp = ds4_gpu_tensor_alloc(index_comp_bytes);
    t_scores = ds4_gpu_tensor_alloc(scores_bytes);
    t_ref = ds4_gpu_tensor_alloc(selected_bytes);
    t_fused = ds4_gpu_tensor_alloc(selected_bytes);
    if (!t_q || !t_weights || !t_index_comp || !t_scores || !t_ref || !t_fused) {
        fprintf(stderr, "device allocation failed\n");
        goto out;
    }

    if (!write_tensor(t_q, q, q_bytes, "q") ||
        !write_tensor(t_weights, weights, weights_bytes, "weights") ||
        !write_tensor(t_index_comp, index_comp, index_comp_bytes, "index_comp")) {
        goto out;
    }

    const float scale = 1.0f / sqrtf((float)(N_HEAD * HEAD_DIM));
    if (!ds4_gpu_indexer_score_one_tensor(t_scores, t_q, t_weights, t_index_comp,
                                          N_COMP, N_HEAD, HEAD_DIM, scale)) {
        fprintf(stderr, "indexer score_one failed\n");
        goto out;
    }
    if (!ds4_gpu_indexer_topk_tensor(t_ref, t_scores, N_COMP, 1, TOP_K)) {
        fprintf(stderr, "indexer topk failed\n");
        goto out;
    }
    if (!ds4_gpu_indexer_score_topk_fused_tensor(t_fused, t_q, t_weights, t_index_comp,
                                                 N_COMP, N_HEAD, HEAD_DIM, scale, TOP_K)) {
        fprintf(stderr, "indexer fused score_topk failed\n");
        goto out;
    }
    if (!ds4_gpu_synchronize()) goto out;

    if (!ds4_gpu_tensor_read(t_ref, 0, ref, selected_bytes) ||
        !ds4_gpu_tensor_read(t_fused, 0, fused, selected_bytes)) {
        fprintf(stderr, "selected readback failed\n");
        goto out;
    }

    if (memcmp(ref, fused, (size_t)selected_bytes) != 0) {
        for (uint32_t i = 0; i < TOP_K; i++) {
            if (ref[i] != fused[i]) {
                fprintf(stderr, "topk mismatch at %u: ref=%u fused=%u\n", i, ref[i], fused[i]);
                break;
            }
        }
        goto out;
    }

    puts("cuda indexer fused smoke: OK");
    rc = 0;

out:
    ds4_gpu_tensor_free(t_q);
    ds4_gpu_tensor_free(t_weights);
    ds4_gpu_tensor_free(t_index_comp);
    ds4_gpu_tensor_free(t_scores);
    ds4_gpu_tensor_free(t_ref);
    ds4_gpu_tensor_free(t_fused);
    ds4_gpu_cleanup();
    free(q);
    free(weights);
    free(index_comp);
    free(ref);
    free(fused);
    return rc;
}
