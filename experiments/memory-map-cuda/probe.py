import os,sys,time,json,threading,hashlib,types,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'helpers'))
from prepare import prepare

def main():
 timer=threading.Timer(290,lambda:os._exit(124));timer.daemon=True;timer.start()
 parser=argparse.ArgumentParser();parser.add_argument('--upstream',type=Path,required=True);args=parser.parse_args()
 out=ROOT/'artifacts';out.mkdir(exist_ok=True)
 report={'mode':'faithful compact circuit, Triton Executor2, fp32 CUDA graph, free-running tokens; not stock HF','is_doom':False,'wall_limit_seconds':290,'cuda_allocator_limit_bytes':8*2**30}
 try:
  started=time.perf_counter();manifest=prepare(args.upstream,ROOT/'scratch',64,4096)
  (out/'patch-manifest.json').write_text(json.dumps(manifest,indent=2))
  sys.path.insert(0,str(ROOT/'scratch'))
  import numpy as np
  import torch,triton
  from chipformer.rv import core,ref,spec,fastexec
  from chipformer.rv.asm import assemble
  from chipformer.q35.ir import Head,HD
  from workloads import workloads
  from state import read_ram
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
  torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory)
  report.update({'torch':torch.__version__,'triton':triton.__version__,'gpu':torch.cuda.get_device_name(),'ram_base':spec.RAM_BASE,'ram_bytes':4*spec.RAM_WORDS,'drom_base':spec.DROM_BASE,'drom_bytes':4*spec.DROM_WORDS,'rom_base':spec.ROM_BASE,'rom_bytes':4*spec.ROM_WORDS})
  print('Building actual64KiB RAM/4MiB DROM graph',flush=True);c=core.build()
  report['graph']={'chip_layers':c.n_layers(),'dims':len(c.dims),'key_heads':sum(isinstance(n,Head) for n in c.nodes),'logical_value_heads':sum(len(n.vheads()) for n in c.nodes if isinstance(n,Head))}
  report['graph']['logical_fp32_recurrent_bytes']=report['graph']['logical_value_heads']*HD*HD*4
  name,assembly,expected_fault,expected_output=workloads(spec.RAM_BASE,4*spec.RAM_WORDS,spec.DROM_BASE,4*spec.DROM_WORDS)[0]
  program=assemble(assembly);rodata=bytearray(4*spec.DROM_WORDS)
  drom_words=[(0,0x80017f80),(len(rodata)//2,0xaabbccdd),(len(rodata)-4,0x12345678)]
  for offset,word in drom_words:rodata[offset:offset+4]=word.to_bytes(4,'little')
  program.rodata=bytes(rodata)
  boot_words=[(4,0x1234abcd),(2*spec.RAM_WORDS+64,0x89abcdef),(4*spec.RAM_WORDS-8,0x55aa1234)]
  def machine():
   result=ref.Machine.from_program(program)
   for offset,word in boot_words:result.ram[offset//4]=word
   return result
  reference=machine();wanted=reference.run(1000)[1:];assert reference.halted and reference.fault is None
  core.snapshot(c,machine());src=Path(fastexec.__file__).read_text()
  patched=src.replace('nf, nb = len(C.OUT_NAMES), C.NPAY','output_names = list(S.FAMILIES) + S.CONTROL_NAMES\n        nf, nb = len(output_names), C.NPAY').replace('enumerate(C.OUT_NAMES)','enumerate(output_names)');assert src!=patched
  (out/'fastexec-standalone.py').write_text(patched)
  module=types.ModuleType('fastexec_standalone_probe');exec(compile(patched,str(out/'fastexec-standalone.py'),'exec'),module.__dict__)
  print('Building Executor2',json.dumps(report['graph']),flush=True)
  fr=module.FreeRun(c,1,'cuda',compile_=False,v2=True);fr.load(0,spec.tok('GO'))
  report['setup_seconds']=time.perf_counter()-started
  with torch.no_grad():
   t=time.perf_counter();first=fr.run(1)[0].tolist();assert first==wanted[:1];report['capture_warmup_seconds']=time.perf_counter()-t
   fr.load(0,spec.tok('GO'));torch.cuda.synchronize();t=time.perf_counter();generated=fr.run(len(wanted))[0].tolist();report['generation_seconds_one_run']=time.perf_counter()-t
  assert generated==wanted,(generated,wanted)
  output=bytes(spec.family(t)[1] for t in generated if spec.family(t)[0]=='BYTE').decode('latin-1');assert output==expected_output
  print('Extracting actual recurrent RAM',flush=True);t=time.perf_counter();actual_ram=read_ram(fr.ex,spec.RAM_BASE,4*spec.RAM_WORDS);report['ram_extraction_seconds']=time.perf_counter()-t
  expected_ram=np.asarray(reference.ram,dtype='<u4').tobytes();assert actual_ram==expected_ram
  trace=np.asarray(generated,dtype='<u4').tobytes();(out/'trace.u32').write_bytes(trace);(out/'actual-ram.bin').write_bytes(actual_ram);(out/'expected-ram.bin').write_bytes(expected_ram);(out/'workload.S').write_text(assembly)
  (out/'input-data.json').write_text(json.dumps({'drom_nonzero_words_offset_value':drom_words,'boot_ram_words_offset_value':boot_words,'drom_sha256':hashlib.sha256(rodata).hexdigest(),'case_name':name},indent=2))
  report.update({'generated_tokens':len(generated),'mismatches':0,'output':output,'halted':True,'reference_tokens_used_as_inputs':False,'ram_bytes_compared':len(actual_ram),'ram_byte_mismatches':0,'actual_ram_sha256':hashlib.sha256(actual_ram).hexdigest(),'trace_sha256':hashlib.sha256(trace).hexdigest(),'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'total_seconds':time.perf_counter()-started,'timed_work':'FreeRun CUDA graph replay, self-fed next-token embeddings, trace copies and final CPU transfer; excludes setup, graph capture, reference generation and RAM extraction','source_adaptation':'Only merged-chat Q family omitted from standalone FreeRun embedding table; circuit memory-map patches listed in patch-manifest.json'})
 except Exception as e:
  import traceback
  report['error']=repr(e);report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
 finally:
  (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True);timer.cancel()
 if 'error' in report:sys.exit(1)

if __name__=='__main__':main()
