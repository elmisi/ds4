#!/usr/bin/env python3
"""Isolated sm_75 token-tile regression audit; run only on an idle Ampere+ GPU.

Use the operational quota/resource runner. This builds in --output, never
replaces project objects, and expects the unguarded version to fail its oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-ref', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nvcc', default='/usr/local/cuda/bin/nvcc')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve() / 'arch-audit'
    output.mkdir(parents=True, exist_ok=False)
    before = subprocess.check_output(['git', 'show', args.before_ref + ':ds4_cuda.cu'], cwd=root)
    after = (root / 'ds4_cuda.cu').read_bytes()
    flags = ['-O3', '-g', '-lineinfo', '--use_fast_math', '-arch=sm_75',
             '-Xcompiler', '-march=native', '-Xcompiler', '-pthread', '-I' + str(root)]
    mmq = ['ds4_ggml_stubs', 'ds4_mmq', 'ds4_mmq_d2r', 'quantize', 'mmid', 'mmvq', 'ds4_repack']
    objects = [root / 'tests/test_deepseek41_cuda.o', root / 'ds4_image.o']
    objects += [root / 'cuda/mmq' / (name + '.o') for name in mmq]
    if any(not path.is_file() for path in objects):
        raise RuntimeError('build native tests/test_deepseek41_cuda prerequisites first')
    results = {}
    for label, source in [('before', before), ('after', after)]:
        src, obj, binary = (output / (label + suffix) for suffix in ['.cu', '.o', '-test'])
        src.write_bytes(source)
        compile_command = [args.nvcc, *flags, '-c', str(src), '-o', str(obj)]
        link_command = [args.nvcc, *flags, '-o', str(binary), str(obj),
                        *map(str, objects), '-lm', '-lcudart', '-lcublas']
        with (output / (label + '-build.log')).open('w') as log:
            for command in [compile_command, link_command]:
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        with (output / (label + '-test.log')).open('w') as log:
            test = subprocess.run([str(binary), '--tp-attention'], cwd=root,
                                  stdout=log, stderr=subprocess.STDOUT)
        text = (output / (label + '-test.log')).read_text()
        results[label] = dict(returncode=test.returncode,
                              source_sha256=hashlib.sha256(source).hexdigest(),
                              compile_command=compile_command, link_command=link_command,
                              log_tail=text.splitlines()[-8:])
        print(json.dumps({label: results[label]}), flush=True)
    passed = (results['before']['returncode'] != 0 and
              results['after']['returncode'] == 0 and
              'isfinite(got[d]) && err < 3e-5' in (output / 'before-test.log').read_text())
    # A nonzero exit alone is not evidence of the targeted numerical bug.
    results['expected_regression_reproduced_and_fixed'] = passed
    (output / 'result.json').write_text(json.dumps(results, indent=2) + '\n')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
