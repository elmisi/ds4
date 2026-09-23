#!/usr/bin/env python3
"""Exercise the real Makefile stamp without compiling CUDA or touching a build."""
from pathlib import Path
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    source = (root / 'Makefile').read_text()
    block = source.split('# BEGIN CUDA CONFIG STAMP', 1)[1].split('\n', 1)[1]
    block = block.split('# END CUDA CONFIG STAMP', 1)[0]
    with tempfile.TemporaryDirectory(prefix='ds4-cuda-config-test-') as directory:
        cwd = Path(directory)
        names = ['ds4_cuda.o', 'mmq.o', 'repack.o']
        (cwd / 'unit.cu').touch()
        # Source prerequisites precede the stamp exactly as in the actual file.
        (cwd / 'Makefile').write_text(
            'MMQ_OBJS := mmq.o repack.o\n'
            'all: ds4_cuda.o $(MMQ_OBJS)\n'
            'ds4_cuda.o $(MMQ_OBJS): unit.cu\n'
            '\t@test "$<" = unit.cu\n'
            '\t@echo "$@" >> built.log\n'
            '\t@touch "$@"\n' + block)

        def build(flags='-arch=sm_121a', compiler='nvcc', includes='-Icuda/mmq'):
            log = cwd / 'built.log'
            log.write_text('')
            subprocess.run(['make', '-s', '-j2', 'all', f'NVCC={compiler}',
                            f'NVCCFLAGS={flags}', f'MMQ_INCLUDES={includes}'],
                           cwd=cwd, check=True)
            return sorted(log.read_text().splitlines())

        expected = sorted(names)
        assert build() == expected, 'first build must compile every object'
        assert build() == [], 'unchanged configuration must reuse every object'
        assert build('-arch=sm_75') == expected, 'arch switch must rebuild all'
        assert build('-arch=sm_75') == [], 'unchanged alternate arch must reuse'
        assert build() == expected, 'switching back must also rebuild all'
        assert build('-arch=sm_121a -O2') == expected, 'flags must invalidate'
        assert build(compiler='/opt/cuda/bin/nvcc') == expected
        assert build(includes="-Ipath'with-quote") == expected
        assert build(includes="-Ipath'with-quote") == []
        assert not list(cwd.glob('.ds4-cuda-build-config.*')), 'temporary leak'
    print('CUDA build configuration regression tests passed')


if __name__ == '__main__':
    main()
