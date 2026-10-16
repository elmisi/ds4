#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parents[2]
folder=Path(__file__).resolve().parent/'validation-20261016'
obj=folder/'obj';bin=folder/'bin'
flags=['-O3','-ffast-math','-g','-march=native','-Wall','-Wextra','-std=c99','-D_GNU_SOURCE','-fno-finite-math-only','-ffunction-sections','-fdata-sections','-I'+str(root)]
def compile(args):
 source=root/args[args.index('-c')+1]
 target=Path(args[args.index('-o')+1])
 dependencies=[source,root/'ds4.h',root/'ds4_kvstore.h']
 if source.name=='ds4_test.c':dependencies.append(root/'ds4_server.c')
 if target.exists() and all(p.stat().st_mtime_ns<=target.stat().st_mtime_ns for p in dependencies):return
 subprocess.run(['cc',*flags,*args],cwd=root,check=True)
with ThreadPoolExecutor(max_workers=2) as pool:
 list(pool.map(compile,[['-c','ds4_server.c','-o',str(obj/'ds4_server.o')],['-c','ds4_kvstore.c','-o',str(obj/'ds4_kvstore.o')],['-DDS4_NO_GPU','-Wno-unused-function','-c','tests/ds4_test.c','-o',str(obj/'host-tests.o')],['-c',str(Path(__file__).with_name('payload_roundtrip.c')),'-o',str(obj/'payload-roundtrip.o')]]))
core=['ds4.o','ds4_image.o','ds4_distributed.o','ds4_tp.o','ds4_ssd.o','ds4_cuda.o','ds4_layer_pack.o','ds4_engram.o']
core=[str(obj/n) for n in core]+[str(p) for p in sorted((obj/'cuda/mmq').glob('*.o'))]
link=['/usr/local/cuda/bin/nvcc','-O3','-g','-lineinfo','--use_fast_math','-gencode','arch=compute_121a,code=sm_121a','-Xcompiler','-march=native','-Xcompiler','-pthread']
libs=['-lm','-Xcompiler','-pthread','-L/usr/local/cuda/targets/sbsa-linux/lib','-L/usr/local/cuda/lib64','-lcudart','-lcublas']
for name,entry in [('ds4-server','ds4_server.o'),('host-tests','host-tests.o'),('payload-roundtrip','payload-roundtrip.o')]:
 command=[*link,'-o',str(bin/name),str(obj/entry),str(obj/'ds4_help.o'),str(obj/'ds4_kvstore.o'),str(obj/'rax.o'),str(obj/'ds4_gpu_args.o'),*core,*libs]
 subprocess.run(command,cwd=root,check=True)
 elf=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--list-elf',str(bin/name)],text=True)
 (folder/(name+'-cubins.txt')).write_text(elf)
 cubins=[line for line in elf.splitlines() if '.cubin' in line]
 assert len(cubins)==9 and all('sm_121a' in line for line in cubins),elf
print(json.dumps({name:hashlib.sha256((bin/name).read_bytes()).hexdigest() for name in ['ds4-server','host-tests']},indent=2))
