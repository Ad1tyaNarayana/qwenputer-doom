import argparse,hashlib,json,os,struct,threading,time
from datetime import datetime,timezone
from pathlib import Path
from runtime import ROOT,ELF_SHA,TRACE_SHA,TOTAL_TOKENS,build,restore_checkpoint
FRAME_SHA='6d60d53e30603c046bc61d6fdc1818cc9318f227064ec9b77285b1bda15d0db2'
FINAL_RAM_SHA='b9ea6e4ad1a05162f571970604e1a085078d8cf6ae21da48749b47f4dd80b3ef'

def replace_shared(temporary,path):
 for attempt in range(100):
  try:
   temporary.replace(path);return
  except PermissionError:
   if attempt==99:raise
   time.sleep(0.05)

def atomic_json(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));replace_shared(temp,path)

def save_state(fr,offset,path,compute_seconds):
 import torch
 heads=[s for s in fr.ex.steps if s['kind']=='head'];assert torch.count_nonzero(fr.ex.y).item()==0
 data={'elf_sha256':ELF_SHA,'offset':offset,'x':fr.x.detach().cpu().clone(),'tok':fr.tok.detach().cpu().clone(),'heads':[{'S':s['S'].detach().cpu().clone(),'hist':s['hist'].detach().cpu().clone()} for s in heads],'compute_seconds':compute_seconds}
 temp=path.with_suffix('.tmp');torch.save(data,temp);replace_shared(temp,path)

def draw_strip(frame,coverage,descriptor,palette,indices):
 import numpy as np
 d=descriptor;mul=2 if d['flags']&1 else 1
 assert 0<=d['x']<320 and 0<=d['y']<240 and 0<d['h']<=240-d['y']
 assert 0<d['n']<=320 and d['x']+d['n']*mul<=320 and d['flags'] in [0,1]
 assert len(indices)==d['n']*d['h'] and len(palette)==512
 values=np.frombuffer(palette,dtype='<u2')[np.frombuffer(indices,dtype='u1')].reshape(d['n'],d['h']).T
 values=np.repeat(values,mul,axis=1)
 target=(slice(d['y'],d['y']+d['h']),slice(d['x'],d['x']+d['n']*mul))
 assert not coverage[target].any()
 frame[target]=values;coverage[target]=True

def write_preview(frame,path):
 import numpy as np
 from PIL import Image
 rgb=np.empty((240,320,3),dtype='u1');rgb[:,:,0]=((frame>>11)&31)*255//31;rgb[:,:,1]=((frame>>5)&63)*255//63;rgb[:,:,2]=(frame&31)*255//31
 Image.fromarray(rgb).save(path)

def main():
 p=argparse.ArgumentParser();p.add_argument('--max-seconds',type=float,default=7200);p.add_argument('--stop-after-tokens',type=int);args=p.parse_args()
 timer=threading.Timer(args.max_seconds,lambda:os._exit(124));timer.daemon=True;timer.start()
 out=ROOT/'private-run';strips_dir=out/'gpu-strips';strips_dir.mkdir(exist_ok=True)
 report={'phase':'full-frame','full_frame_completed':False,'wall_limit_seconds':args.max_seconds}
 try:
  import numpy as np,torch
  from PIL import Image
  fr,expected,spec,base=build();report.update(base)
  sys_state=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True)
  offset=restore_checkpoint(fr,out/'checkpoint.pt')
  compute_seconds=float(sys_state.get('compute_seconds',json.loads((out/'prefix-report.json').read_text())['compute_seconds']))
  initial_offset=offset;started=time.perf_counter();last_checkpoint=offset
  trace_path=out/'actual.tokens.u32';assert trace_path.stat().st_size>=offset*4
  with trace_path.open('r+b') as f:f.truncate(offset*4)
  saved_actual=np.memmap(trace_path,mode='r',dtype='<u4');assert np.array_equal(saved_actual,expected[:offset]);del saved_actual
  manifest=json.loads((ROOT/'inputs/strips.json').read_text());assert manifest['elf_sha256']==ELF_SHA
  markers=manifest['strips'];addresses=manifest['metadata_addresses'];assert len(markers)==25
  frame=np.zeros((240,320),dtype='<u2');coverage=np.zeros((240,320),dtype=bool);completed=[]
  for marker in markers:
   if marker['generated_token_offset']>offset:continue
   strip_path=strips_dir/f"{marker['serial']:02d}.npz"
   with np.load(strip_path,allow_pickle=False) as saved:
    descriptor=json.loads(str(saved['descriptor']));draw_strip(frame,coverage,descriptor,saved['palette'].tobytes(),saved['indices'].tobytes())
   completed.append(descriptor)
  print(f'Resumed actual GPU checkpoint at {offset}, {len(completed)} saved strips',flush=True)
  from state import read_ram
  boundaries=[m['generated_token_offset'] for m in markers if m['generated_token_offset']>offset]+[TOTAL_TOKENS]
  for target in boundaries:
   with trace_path.open('ab') as trace,torch.no_grad():
    while offset<target:
     if args.stop_after_tokens and offset>=args.stop_after_tokens:break
     amount=min(8192,target-offset)
     if args.stop_after_tokens:amount=min(amount,args.stop_after_tokens-offset)
     t=time.perf_counter();generated=fr.run(amount)[0];compute_seconds+=time.perf_counter()-t
     wanted=expected[offset:offset+amount]
     if not np.array_equal(generated,wanted):
      index=int(np.flatnonzero(generated!=wanted)[0]);raise AssertionError({'position':offset+index,'actual':int(generated[index]),'expected':int(wanted[index])})
     trace.write(generated.astype('<u4').tobytes());trace.flush();offset+=amount
     progress={'phase':'full-frame','generated_tokens':offset,'total_tokens':TOTAL_TOKENS,'mismatches':0,'compute_seconds_total':compute_seconds,'elapsed_seconds_this_process':time.perf_counter()-started,'gpu_strips_extracted':len(completed),'written_pixels':int(coverage.sum()),'checkpoint_offset':last_checkpoint,'full_frame_completed':False,'updated_utc':datetime.now(timezone.utc).isoformat()}
     atomic_json(out/'progress.json',progress)
     with (out/'chunks.jsonl').open('a') as log:log.write(json.dumps({'end_offset':offset,'amount':amount,'compute_seconds_total':compute_seconds,'sha256':hashlib.sha256(generated.astype('<u4').tobytes()).hexdigest()})+'\n')
     print(json.dumps(progress),flush=True)
     if offset-last_checkpoint>=262144 and offset!=target:
      save_state(fr,offset,out/'checkpoint.pt',compute_seconds);last_checkpoint=offset
   if args.stop_after_tokens and offset<target:break
   if target!=TOTAL_TOKENS:
    marker=next(m for m in markers if m['generated_token_offset']==target);assert int(fr.tok[0].item())==spec.tok('BYTE',256)
    ram=read_ram(fr.ex,spec.RAM_BASE,4*spec.RAM_WORDS)
    descriptor={name:struct.unpack_from('<I',ram,addresses['qp_strip_'+name]-spec.RAM_BASE)[0] for name in ['ptr','x','y','h','n','flags','serial']}
    assert descriptor=={k:marker[k] for k in descriptor}
    start=descriptor['ptr']-spec.RAM_BASE;count=descriptor['n']*descriptor['h'];assert 0<=start<=len(ram)-count
    indices=ram[start:start+count];palette_offset=addresses['pump_lut']-spec.RAM_BASE;palette=ram[palette_offset:palette_offset+512]
    draw_strip(frame,coverage,descriptor,palette,indices)
    np.savez(strips_dir/f"{descriptor['serial']:02d}.npz",descriptor=json.dumps(descriptor),indices=np.frombuffer(indices,dtype='u1'),palette=np.frombuffer(palette,dtype='u1'))
    completed.append(descriptor);write_preview(frame,out/'partial-frame.png')
    atomic_json(out/'gpu-strips.json',{'kind':'Descriptors extracted from actual GPU recurrent RAM at generated marker boundaries','strips':completed,'written_pixels':int(coverage.sum()),'all_pixels_from_gpu_state':True,'reference_pixels_used':False})
    print(f'Extracted GPU strip {descriptor["serial"]}/25 at token {offset}',flush=True)
   save_state(fr,offset,out/'checkpoint.pt',compute_seconds);last_checkpoint=offset
  if offset<TOTAL_TOKENS:
   save_state(fr,offset,out/'checkpoint.pt',compute_seconds)
   report.update({'stop_reason':'requested_token_boundary','checkpoint_offset':offset,'generated_tokens':offset,'gpu_strips_extracted':len(completed),'full_frame_completed':False,'compute_seconds_total':compute_seconds})
  else:
   assert int(fr.tok[0].item())==spec.tok('HALT') and len(completed)==25 and coverage.all()
   raw=frame.astype('<u2').tobytes();actual_frame_sha=hashlib.sha256(raw).hexdigest();assert actual_frame_sha==FRAME_SHA
   (out/'frame.rgb565').write_bytes(raw);write_preview(frame,out/'frame.png')
   actual_ram=read_ram(fr.ex,spec.RAM_BASE,4*spec.RAM_WORDS);(out/'final-gpu-ram.bin').write_bytes(actual_ram);actual_ram_sha=hashlib.sha256(actual_ram).hexdigest();assert actual_ram_sha==FINAL_RAM_SHA
   actual_trace_sha=hashlib.sha256(trace_path.read_bytes()).hexdigest();assert actual_trace_sha==TRACE_SHA
   report.update({'generated_tokens':offset,'mismatches':0,'halted':True,'gpu_strips_extracted':len(completed),'written_pixels':int(coverage.sum()),'full_frame_completed':True,'frame_sha256':actual_frame_sha,'unique_rgb565_colors':int(len(np.unique(frame))),'final_gpu_ram_sha256':actual_ram_sha,'actual_trace_sha256':actual_trace_sha,'compute_seconds_total':compute_seconds,'compute_seconds_this_process':compute_seconds-float(sys_state.get('compute_seconds',json.loads((out/'prefix-report.json').read_text())['compute_seconds'])),'wall_seconds_this_process':time.perf_counter()-started,'started_from_checkpoint_offset':initial_offset,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'reference_pixels_used':False,'pixels_source':'GPU recurrent RAM strip buffers and palette, read at actual generated BYTE256 boundaries','timed_work':'Sum of successful FreeRun CUDA graph generation chunks, including device token decoding, self-fed embedding, trace copies and CPU transfer; excludes setup, graph capture, comparisons, IO, checkpointing and pixel extraction'})
 except Exception as e:
  import traceback
  report['error']=repr(e);report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
 finally:
  atomic_json(out/'full-report.json',report);print(json.dumps(report,indent=2),flush=True);timer.cancel()
 if 'error' in report:raise SystemExit(1)

if __name__=='__main__':main()
