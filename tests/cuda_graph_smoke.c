/* Smoke test for the CUDA Graph capture/replay scaffolding added to the
 * ds4_gpu.h API. Verifies that:
 *   1. Operations issued between capture_begin and capture_end are recorded
 *      but NOT executed (state observed after end_capture must still reflect
 *      pre-capture state).
 *   2. ds4_gpu_graph_launch executes the recorded operations.
 *   3. The handle can be launched multiple times and produces deterministic
 *      results (graph replay overwrites any prior content).
 *
 * The test does not exercise any DS4 inference path, only the raw API
 * contract. Failures here indicate the scaffolding is broken before any
 * decode-tape integration is attempted.
 */

#include "ds4_gpu.h"

#include <stdio.h>
#include <stdlib.h>

#define N 64

static int read_first_value(ds4_gpu_tensor *t, float *out) {
    return ds4_gpu_tensor_read(t, 0, out, sizeof(*out));
}

static int check_capture_does_not_execute(void) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(N * sizeof(float));
    if (!t) {
        fprintf(stderr, "tensor alloc failed\n");
        return 1;
    }
    int rc = 1;
    float observed = -1.0f;

    if (!ds4_gpu_tensor_fill_f32(t, 1.0f, N)) goto out;
    if (!ds4_gpu_synchronize()) goto out;
    if (!read_first_value(t, &observed) || observed != 1.0f) {
        fprintf(stderr, "pre-capture fill: expected 1.0, got %f\n", observed);
        goto out;
    }

    if (!ds4_gpu_graph_capture_begin()) {
        fprintf(stderr, "capture begin failed\n");
        goto out;
    }
    if (!ds4_gpu_tensor_fill_f32(t, 2.0f, N)) {
        fprintf(stderr, "fill during capture failed\n");
        goto out;
    }
    ds4_gpu_graph_handle *h = ds4_gpu_graph_capture_end();
    if (!h) {
        fprintf(stderr, "capture end failed\n");
        goto out;
    }

    /* Capture should not have executed the kernel. The tensor must still
     * hold the pre-capture value. */
    if (!ds4_gpu_synchronize()) goto out_h;
    if (!read_first_value(t, &observed) || observed != 1.0f) {
        fprintf(stderr, "post-capture: expected 1.0 (captured fill not run), got %f\n", observed);
        goto out_h;
    }

    /* Launch the recorded graph and observe the new value. */
    if (!ds4_gpu_graph_launch(h)) goto out_h;
    if (!ds4_gpu_synchronize()) goto out_h;
    if (!read_first_value(t, &observed) || observed != 2.0f) {
        fprintf(stderr, "post-launch: expected 2.0, got %f\n", observed);
        goto out_h;
    }

    /* Reset tensor and re-launch to verify the handle is reusable. */
    if (!ds4_gpu_tensor_fill_f32(t, 3.0f, N)) goto out_h;
    if (!ds4_gpu_synchronize()) goto out_h;
    if (!read_first_value(t, &observed) || observed != 3.0f) {
        fprintf(stderr, "reset before relaunch: expected 3.0, got %f\n", observed);
        goto out_h;
    }
    if (!ds4_gpu_graph_launch(h)) goto out_h;
    if (!ds4_gpu_synchronize()) goto out_h;
    if (!read_first_value(t, &observed) || observed != 2.0f) {
        fprintf(stderr, "relaunch: expected 2.0, got %f\n", observed);
        goto out_h;
    }

    rc = 0;
out_h:
    ds4_gpu_graph_handle_free(h);
out:
    ds4_gpu_tensor_free(t);
    return rc;
}

int main(void) {
    if (!ds4_gpu_init()) {
        fprintf(stderr, "ds4_gpu_init failed\n");
        return 1;
    }
    if (!ds4_gpu_graph_capture_supported()) {
        fprintf(stderr, "graph capture not supported on this backend; skipping\n");
        ds4_gpu_cleanup();
        return 0;
    }
    int rc = check_capture_does_not_execute();
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda graph smoke: OK");
    return rc;
}
