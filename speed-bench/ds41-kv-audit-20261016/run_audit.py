#!/usr/bin/env python3
"""Reproduce the observed scheduler gap on CPU; test an unapplied patch copy."""
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE_REF = "0ce0033fd53f93f0d29d16e6fc48f34892c5159b"


def function(text, name):
    match = re.search(r'^static (?:int|void) ' + re.escape(name) + r'\(', text, re.M)
    if not match:
        raise ValueError(f'missing function: {name}')
    start = text.index('{', match.start())
    depth = 1
    end = start + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[match.start():end] + '\n'


def main():
    original = {name: subprocess.check_output(
        ['git', 'show', BASELINE_REF+':'+name], cwd=ROOT, text=True)
        for name in ('ds4_kvstore.c', 'ds4_server.c')}
    applied = (ROOT/'ds4_kvstore.c').read_text() != original['ds4_kvstore.c']
    candidate = dict(original)
    before = ('    if (live_tokens % step != 0) return 0;\n'
              '    if (live_tokens <= kc->continued_last_store_tokens) return 0;')
    after = ('    /* Backend-safe callbacks can skip an exact interval boundary.\n'
             '     * Keep the interval grid, but allow the first valid frontier\n'
             '     * after crossing it. Never round a live payload backwards. */\n'
             '    if (live_tokens / step <= kc->continued_last_store_tokens / step) return 0;')
    assert candidate['ds4_kvstore.c'].count(before) == 1
    candidate['ds4_kvstore.c'] = candidate['ds4_kvstore.c'].replace(before, after)
    helper = function(original['ds4_server.c'], 'kv_cache_maybe_store_continued')
    before_guard = '    if (!s || !slot) return;'
    after_guard = ('    /* A DS41 prefill callback can precede final logits. */\n'
                   '    if (!s || !slot || !ds4_session_checkpoint_valid(slot->session)) return;')
    assert helper.count(before_guard) == 1
    candidate['ds4_server.c'] = candidate['ds4_server.c'].replace(
        helper, helper.replace(before_guard, after_guard), 1)
    patch = ''.join(''.join(difflib.unified_diff(original[name].splitlines(True),
                  candidate[name].splitlines(True), fromfile='a/'+name, tofile='b/'+name))
                  for name in original)
    if not (HERE/'proposed-fix.patch').exists():
        (HERE/'proposed-fix.patch').write_text(patch)
    results = {}
    with tempfile.TemporaryDirectory(prefix='.build-', dir=HERE) as temporary:
        build = Path(temporary)
        for variant, source in [('baseline', original), ('candidate', candidate)]:
            folder = build/variant
            folder.mkdir()
            (folder/'ds4_kvstore.c').write_text(source['ds4_kvstore.c'])
            helpers = ''.join(function(source['ds4_server.c'], name) for name in
                              ('kv_cache_slot_continued_target', 'kv_cache_slot_note_store',
                               'kv_cache_maybe_store_continued'))
            (folder/'server_helpers.inc').write_text(helpers)
            command = ['cc', '-std=c11', '-D_POSIX_C_SOURCE=200809L', '-O1', '-g',
                       '-ffunction-sections', '-fdata-sections', '-fsanitize=undefined',
                       '-fno-sanitize-recover=all', '-I'+str(ROOT), '-I'+str(folder)]
            if variant == 'candidate':
                command += ['-DAUDIT_CANDIDATE=1']
            command += [str(HERE/'policy_probe.c'), str(folder/'ds4_kvstore.c'),
                        '-Wl,--gc-sections', '-lm', '-o', str(folder/'probe')]
            subprocess.run(command, check=True, cwd=ROOT)
            result = subprocess.check_output([str(folder/'probe')], text=True, cwd=ROOT)
            results[variant] = json.loads(result)
    report = {'status':'PASS', 'kind':'host-only policy reproduction', 'results':results,
              'patch_applied':applied, 'baseline_ref':BASELINE_REF,
              'service_modified':False,
              'source_hashes':{name:hashlib.sha256(text.encode()).hexdigest()
                               for name,text in original.items()},
              'limitations':['GPU state and payload persistence are test doubles.',
                             'GPU serialization, restore equality and latency still need live validation.',
                             'This probe does not rebuild or install the service binary.']}
    output = 'applied-policy-results.json' if applied else 'policy-results.json'
    (HERE/output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
