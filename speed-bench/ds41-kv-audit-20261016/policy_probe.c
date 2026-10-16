/* Host-only audit: use the real KV scheduler and extracted server helpers.
 * The session and payload writer are doubles: this does not validate GPU
 * serialization, numerical equality or disk restore.
 */
#include "ds4_kvstore.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>

typedef ds4_kvstore kv_disk_cache;
struct ds4_session { ds4_tokens tokens; bool valid; };
typedef struct { ds4_session *session; int continued_last_store_tokens; } server_slot;
typedef struct { kv_disk_cache kv; } server;

static int attempts, invalid_attempts, writes, last_write;
const ds4_tokens *ds4_session_tokens(ds4_session *s) { return s ? &s->tokens : NULL; }
bool ds4_session_checkpoint_valid(const ds4_session *s) { return s && s->valid; }
static int kv_cache_continued_store_target(const kv_disk_cache *kc, int n) {
    return ds4_kvstore_continued_store_target(kc, n);
}
static bool kv_cache_store_live_prefix(server *s, server_slot *slot,
                                       const ds4_tokens *tokens, int len,
                                       const char *reason) {
    (void)s; (void)reason;
    attempts++;
    if (!slot->session->valid) { invalid_attempts++; return false; }
    assert(tokens->len == len);
    writes++;
    last_write = len;
    return true;
}

/* Written by run_audit.py from the source under investigation. */
#include "server_helpers.inc"

static void check_existing_contracts(void) {
    ds4_kvstore kc = { .enabled = true };
    kc.opt = ds4_kvstore_default_options();
    assert(ds4_kvstore_continued_store_target(&kc, 10239) == 0);
    assert(ds4_kvstore_continued_store_target(&kc, 10240) == 10240);
    kc.continued_last_store_tokens = 4096;
    assert(ds4_kvstore_continued_store_target(&kc, 10240) == 10240);
    kc.continued_last_store_tokens = 24576;
    assert(ds4_kvstore_continued_store_target(&kc, 30720) == 30720);
    kc.continued_last_store_tokens = 10240;
    assert(ds4_kvstore_continued_store_target(&kc, 18432) == 0);
    assert(ds4_kvstore_continued_store_target(&kc, 20480) == 20480);
    kc.opt.boundary_align_tokens = 0;
    kc.continued_last_store_tokens = 20480;
    assert(ds4_kvstore_continued_store_target(&kc, 29999) == 0);
    assert(ds4_kvstore_continued_store_target(&kc, 30000) == 30000);
    kc.opt = ds4_kvstore_default_options();
    kc.continued_last_store_tokens = 0;
    int old = ds4_kvstore_suppress_continued_store(&kc, 10240);
    assert(old == 0 && kc.continued_last_store_tokens == 10240);
    assert(ds4_kvstore_continued_store_target(&kc, 10240) == 0);
    ds4_kvstore_restore_suppressed_continued(&kc, old, 10240);
    assert(ds4_kvstore_continued_store_target(&kc, 10240) == 10240);
    kc.enabled = false;
    assert(ds4_kvstore_continued_store_target(&kc, 254896) == 0);
}

int main(void) {
    check_existing_contracts();
    ds4_session session = {0};
    server s = { .kv = { .enabled = true } };
    s.kv.opt = ds4_kvstore_default_options();
    s.kv.opt.continued_interval_tokens = 20480;
    server_slot slot = { .session = &session };
    /* Recorded server callbacks, followed by the valid final-logits callback.
     * ds4.c's DS41 sync invalidates the checkpoint before intermediate callbacks
     * and only validates it after computing the final output head.
     */
    const int frontier[] = {32768, 65536, 98304, 131072, 163840,
                            196608, 229376, 254896};
    for (unsigned i = 0; i < sizeof(frontier)/sizeof(frontier[0]); i++) {
        session.tokens.len = frontier[i];
        session.valid = false;
        kv_cache_maybe_store_continued(&s, &slot);
    }
    session.valid = true;
    kv_cache_maybe_store_continued(&s, &slot);
    kv_cache_maybe_store_continued(&s, &slot); /* same final frontier: no duplicate */
#ifdef AUDIT_CANDIDATE
    assert(writes == 1 && last_write == 254896);
    assert(attempts == 1 && invalid_attempts == 0);
    /* A restored non-aligned checkpoint waits until the next interval bucket. */
    ds4_kvstore kc = s.kv;
    kc.continued_last_store_tokens = 254896;
    assert(ds4_kvstore_continued_store_target(&kc, 254896) == 0);
    assert(ds4_kvstore_continued_store_target(&kc, 260000) == 0);
    assert(ds4_kvstore_continued_store_target(&kc, 266240) == 266240);
    kc.continued_last_store_tokens = INT_MAX - 20480;
    assert(ds4_kvstore_continued_store_target(&kc, INT_MAX) == INT_MAX);
#else
    assert(writes == 0 && attempts == 1 && invalid_attempts == 1);
#endif
    printf("{\"writes\":%d,\"attempts\":%d,\"invalid_attempts\":%d,"
           "\"last_write\":%d,\"existing_policy_contracts\":\"PASS\"}\n",
           writes, attempts, invalid_attempts, last_write);
    return 0;
}
