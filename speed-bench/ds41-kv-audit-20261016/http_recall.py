import datetime, hashlib, http.client, importlib.util, json, os, subprocess, threading, time
from pathlib import Path

root=Path(__file__).resolve().parent
state=root/'validation-20261016'
run=Path(state)/'http-recall-254k'
run.mkdir()
fixture=root/'fixtures/recall-256k.txt'
assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='28c641b634fd1156823582d0eca7aa276a8db032c137124f79da9804182e2e00'
spec=importlib.util.spec_from_file_location('recall_validator',root/'check_recall.py')
validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
pid=int(subprocess.check_output(['systemctl','--user','show','ds4-ds41.service','-p','MainPID','--value'],text=True))
assert pid>0
payload={'model':'deepseek-v4.1-flash','messages':[{'role':'user','content':fixture.read_text()}],'temperature':0,'top_p':1,'min_p':0.05,'seed':12345,'reasoning_effort':'none','max_tokens':512,'stream':False}
body=json.dumps(payload).encode()
(run/'request.json').write_bytes(body)
(run/'provenance.json').write_text(json.dumps({'start':datetime.datetime.now().astimezone().isoformat(),'pid':pid,'expert_cache_gib':80,'kv_budget_mib':65536,'minimum_available_gib':8,'fixture':str(fixture),'request_sha256':hashlib.sha256(body).hexdigest(),'swap_enabled':bool(Path('/proc/swaps').read_text().splitlines()[1:]),'service_command':Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()},indent=2)+'\n')

def counters(path):
 values={}
 for line in Path(path).read_text().splitlines():
  key,sep,value=line.partition(':')
  if sep:
   try:values[key]=int(value.split()[0])
   except (ValueError,IndexError):pass
 return values

results=[]
for attempt in (1,2):
 trial=run/f'request-{attempt}'
 trial.mkdir()
 box={};done=threading.Event();started=time.monotonic()
 def request():
  try:
   conn=http.client.HTTPConnection('127.0.0.1',8000,timeout=2400)
   conn.request('POST','/v1/chat/completions',body=body,headers={'Content-Type':'application/json'})
   response=conn.getresponse();raw=response.read();(trial/'response.json').write_bytes(raw)
   box.update(status=response.status,response=json.loads(raw))
   conn.close()
  except BaseException as exc:box['error']=repr(exc)
  finally:done.set()
 thread=threading.Thread(target=request,daemon=True);thread.start()
 samples=[];low=0;failure=None;last_print=0
 with (trial/'resources.jsonl').open('x') as resources:
  while True:
   try:
    mem=counters('/proc/meminfo');proc=counters(f'/proc/{pid}/status')
   except OSError:
    failure='service process disappeared';break
   current={'elapsed_s':round(time.monotonic()-started,3),'available_kib':mem['MemAvailable'],'process_swap_kib':proc.get('VmSwap',0),'rss_kib':proc.get('VmRSS',0)}
   samples.append(current);resources.write(json.dumps(current)+'\n');resources.flush()
   low=low+1 if current['available_kib']<8*2**20 else 0
   if low>=3:failure='memory-floor';break
   if current['process_swap_kib'] or mem.get('SwapTotal',0):failure='swap unexpectedly enabled';break
   if current['elapsed_s']>(2400 if attempt==1 else 120):failure='timeout';break
   if done.is_set():break
   if current['elapsed_s']-last_print>=30:
    print(f'REQUEST {attempt} elapsed={current["elapsed_s"]:.0f}s available={current["available_kib"]/2**20:.2f}GiB',flush=True)
    last_print=current['elapsed_s']
   done.wait(2)
 summary={'attempt':attempt,'wall_seconds':time.monotonic()-started,'minimum_available_gib':min(s['available_kib'] for s in samples)/2**20,'peak_process_swap_kib':max(s['process_swap_kib'] for s in samples),'stop_reason':failure,'returncode':0 if failure is None and box.get('status')==200 else 1}
 if failure:
  subprocess.run(['dgx-ctl','stop','ds4-ds41'],check=False)
  summary['returncode']=1
  (trial/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  raise RuntimeError(failure)
 if box.get('error') or box.get('status')!=200:
  summary['error']=box
  (trial/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  raise RuntimeError(str(box))
 response=box['response'];choice=response['choices'][0];content=choice['message'].get('content') or ''
 (trial/'content-original.txt').write_text(content)
 normalized=content.rstrip('\n')+'\n'
 (trial/'stdout.log').write_text(normalized)
 summary.update(usage=response.get('usage'),finish_reason=choice.get('finish_reason'),trailing_newline_normalized=normalized!=content)
 (trial/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 ok,reason,digest=validator.check(trial)
 prompt_tokens=response.get('usage',{}).get('prompt_tokens',0)
 assert ok,reason
 assert 250000<=prompt_tokens<262000,prompt_tokens
 assert choice.get('finish_reason')=='stop',choice.get('finish_reason')
 summary.update(recall='16/16',stdout_sha256=digest)
 if attempt==2:
  assert response['usage']['prompt_tokens_details']['cached_tokens']==prompt_tokens, response['usage']
  assert results[0]['stdout_sha256']==digest, 'repeat output changed'
 (trial/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 results.append(summary)
 print('PASS',json.dumps(summary),flush=True)
(run/'summary.json').write_text(json.dumps({'status':'PASS','requests':results},indent=2)+'\n')
print('COMPLETE',run,flush=True)
