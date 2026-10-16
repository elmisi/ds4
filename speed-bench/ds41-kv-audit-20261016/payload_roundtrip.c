#define _POSIX_C_SOURCE 200809L
#include "ds4.h"
#include "ds4_kvstore.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { CTX = 262144, FRONTIER = 254000, STEPS = 256, APPEND = 4096 };
static char error[512];
#define REQUIRE(condition, message) do { if (!(condition)) { \
    fprintf(stderr, "FAIL: %s (%s)\n", message, error); goto done; } } while (0)
static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static void progress(void *unused, const char *event, int current, int total) {
    (void)unused;
    if (!strcmp(event, "prefill_chunk") && (current == total || current % 32768 == 0))
        fprintf(stderr, "prefill %d/%d\n", current, total);
}
static void log_cache(void *unused, ds4_kvstore_log_type type, const char *message) {
    (void)unused; (void)type; fprintf(stderr, "%s\n", message);
}
static int finite_vector(const float *v, int n) {
    for (int i = 0; i < n; i++) if (!isfinite(v[i])) return 0;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s MODEL ENGRAM CORPUS CACHE_DIR\n", argv[0]); return 2;
    }
    int rc = 1, vocab = 0, checks = 0;
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens tokens = {0}, effective = {0};
    ds4_kvstore cache = {0};
    ds4_kvstore_load_result loaded = {0};
    char *text = NULL, *key = NULL;
    float *reference = NULL, *actual = NULL;
    float expected_logprobs[STEPS];
    FILE *input = NULL, *vectors = NULL;
    double prefill_s = 0, save_s = 0, load_s = 0, append_s = 0;
    ds4_engine_options options = {
        .model_path = argv[1], .engram_model_path = argv[2],
        .backend = DS4_BACKEND_CUDA, .n_threads = 8,
        .context_size = CTX, .placement_ctx_hint = CTX, .prefill_chunk = 8192,
        .power_percent = 100, .warm_weights = true, .ssd_streaming = true,
        .ssd_streaming_cache_bytes = 80ull << 30,
    };
    input = fopen(argv[3], "rb");
    REQUIRE(input && fseek(input, 0, SEEK_END) == 0, "open corpus");
    long length = ftell(input);
    REQUIRE(length > 0 && fseek(input, 0, SEEK_SET) == 0, "size corpus");
    text = malloc((size_t)length + 1);
    REQUIRE(text && fread(text, 1, (size_t)length, input) == (size_t)length, "read corpus");
    text[length] = 0; fclose(input); input = NULL;
    REQUIRE(ds4_engine_open(&engine, &options) == 0, "open engine");
    ds4_tokenize_text(engine, text, &tokens);
    REQUIRE(tokens.len >= FRONTIER + APPEND, "enough actual corpus tokens");
    REQUIRE(ds4_session_create(&session, engine, CTX) == 0, "create session");
    ds4_session_set_progress(session, progress, NULL);
    ds4_tokens prefix = {.v = tokens.v, .len = FRONTIER, .cap = tokens.cap};
    double start = now();
    REQUIRE(ds4_session_sync(session, &prefix, error, sizeof(error)) == 0, "fresh prefill");
    prefill_s = now() - start;
    REQUIRE(ds4_session_checkpoint_valid(session), "valid checkpoint after prefill");
    vocab = ds4_engine_vocab_size(engine);
    REQUIRE(vocab > 0, "vocabulary");
    reference = malloc((size_t)(STEPS+1) * vocab * sizeof(float));
    actual = malloc((size_t)vocab * sizeof(float));
    REQUIRE(reference && actual, "allocate reference vectors");
    REQUIRE(ds4_kvstore_open(&cache, argv[4], 65536, false,
                            ds4_kvstore_default_options(), "roundtrip", log_cache, NULL), "open isolated KV");
    cache.opt.continued_interval_tokens = 20480;
    start = now();
    REQUIRE(ds4_kvstore_maybe_store_continued(&cache, engine, session, NULL,
                                            error, sizeof(error)), "store valid off-grid prefix");
    save_s = now() - start;
    REQUIRE(!ds4_kvstore_maybe_store_continued(&cache, engine, session, NULL,
                                             error, sizeof(error)), "no duplicate continued save");
    size_t key_length = 0;
    key = ds4_kvstore_render_tokens_text(engine, &prefix, &key_length);
    REQUIRE(key && key_length > 0, "render exact persisted prefix");
    int entry = ds4_kvstore_find_text_prefix(&cache, key, ds4_engine_model_id(engine),
                                            ds4_engine_routed_quant_bits(engine), CTX);
    REQUIRE(entry >= 0 && cache.len == 1 && cache.entry[entry].tokens == FRONTIER,
            "one discoverable full prompt checkpoint");
    char vector_path[4096];
    REQUIRE(snprintf(vector_path, sizeof(vector_path), "%s/../reference.f32", argv[4])
            < (int)sizeof(vector_path), "vector artifact path");
    vectors = fopen(vector_path, "wbx");
    REQUIRE(vectors, "create exclusive vector artifact");
    fprintf(stderr, "saved actual KV payload; recording %d full-logit steps\n", STEPS);
    for (int i = 0; i <= STEPS; i++) {
        float *v = reference + (size_t)i * vocab;
        REQUIRE(ds4_session_copy_logits(session, v, vocab) == vocab && finite_vector(v, vocab), "fresh finite logits");
        REQUIRE(fwrite(v, sizeof(float), (size_t)vocab, vectors) == (size_t)vocab, "write vector artifact");
        if (i == STEPS) break;
        ds4_token_score score;
        REQUIRE(ds4_session_token_logprob(session, tokens.v[FRONTIER+i], &score) == 1 && isfinite(score.logprob), "fresh logprob");
        expected_logprobs[i] = score.logprob;
        REQUIRE(ds4_session_eval(session, tokens.v[FRONTIER+i], error, sizeof(error)) == 0, "fresh teacher-forced step");
    }
    fclose(vectors); vectors = NULL;
    start = now();
    REQUIRE(ds4_kvstore_try_load_text(&cache, engine, session, key, &effective,
                                     &loaded, NULL, false) == FRONTIER, "restore through actual disk KV reader");
    load_s = now() - start;
    REQUIRE(effective.len == FRONTIER && ds4_session_common_prefix(session, &prefix) == FRONTIER,
            "restored exact prompt tokens");
    fprintf(stderr, "restored disk KV; comparing complete logits and logprobs\n");
    for (int i = 0; i <= STEPS; i++) {
        REQUIRE(ds4_session_copy_logits(session, actual, vocab) == vocab && finite_vector(actual, vocab), "restored finite logits");
        REQUIRE(memcmp(actual, reference + (size_t)i * vocab, (size_t)vocab*sizeof(float)) == 0, "bit-identical restored logits");
        checks++;
        if (i == STEPS) break;
        ds4_token_score score;
        REQUIRE(ds4_session_token_logprob(session, tokens.v[FRONTIER+i], &score) == 1 &&
                memcmp(&score.logprob, &expected_logprobs[i], sizeof(float)) == 0, "bit-identical restored logprob");
        REQUIRE(ds4_session_eval(session, tokens.v[FRONTIER+i], error, sizeof(error)) == 0, "restored teacher-forced step");
    }
    prefix.len = FRONTIER + APPEND;
    start = now();
    REQUIRE(ds4_session_sync(session, &prefix, error, sizeof(error)) == 0, "append to actual 258K");
    append_s = now() - start;
    REQUIRE(ds4_session_pos(session) == FRONTIER + APPEND &&
            ds4_session_common_prefix(session, &prefix) == FRONTIER + APPEND &&
            ds4_session_copy_logits(session, actual, vocab) == vocab && finite_vector(actual, vocab),
            "append position and finite logits");
    printf("{\"status\":\"PASS\",\"frontier\":%d,\"append_frontier\":%d,"
           "\"full_vector_checks\":%d,\"vocabulary\":%d,\"logprobs_equal\":%d,"
           "\"prefill_s\":%.6f,\"save_s\":%.6f,\"load_s\":%.6f,\"append_s\":%.6f}\n",
           FRONTIER, FRONTIER+APPEND, checks, vocab, STEPS, prefill_s, save_s, load_s, append_s);
    rc = 0;
done:
    if (input) fclose(input);
    if (vectors) fclose(vectors);
    ds4_kvstore_load_result_free(&loaded);
    ds4_kvstore_close(&cache);
    ds4_tokens_free(&effective);
    if (session) ds4_session_free(session);
    if (engine) ds4_engine_close(engine);
    ds4_tokens_free(&tokens);
    free(text); free(key); free(reference); free(actual);
    return rc;
}
