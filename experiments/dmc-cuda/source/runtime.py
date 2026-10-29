import hashlib,json,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ELF_SHA='9effb70dfe4cad579f0381a657ac3711564da0278a1410f59789e61fa26ca340'
TRACE_SHA='23d6d7138e2612dfd971f3d86ac3f711c98a0cd2ac90e77410bcc38240edde38'
TOTAL_TOKENS=6342933
sys.path.insert(0,str(ROOT/'helpers'))
from prepare import prepare

def build():
 started=time.perf_counter();out=ROOT/'private-run';out.mkdir(exist_ok=True)
 upstream=ROOT.parents[1]/'.upstream/qwenputer'
 scratch=ROOT/'scratch'
 if not (scratch/'chipformer').exists():
  manifest=prepare(upstream,scratch,64,4096);(out/'patch-manifest.json').write_text(json.dumps(manifest,indent=2))
 sys.path.insert(0,str(scratch))
 import numpy as np
 import torch,triton
 from chipformer.rv import core,elf,ref,spec,fastexec
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory)
 firmware=ROOT/'inputs/doom.elf';assert hashlib.sha256(firmware.read_bytes()).hexdigest()==ELF_SHA
 trace_path=ROOT/'inputs/reference.tokens.u32';assert hashlib.sha256(trace_path.read_bytes()).hexdigest()==TRACE_SHA
 expected=np.memmap(trace_path,mode='r',dtype='<u4');assert len(expected)==TOTAL_TOKENS
 program=elf.program(firmware);c=core.build();core.snapshot(c,ref.Machine.from_program(program))
 source=Path(fastexec.__file__).read_text();patched=source.replace('nf, nb = len(C.OUT_NAMES), C.NPAY','output_names = list(S.FAMILIES) + S.CONTROL_NAMES\n        nf, nb = len(output_names), C.NPAY').replace('enumerate(C.OUT_NAMES)','enumerate(output_names)');assert source!=patched
 (out/'fastexec-standalone.py').write_text(patched);module=types.ModuleType('dmc_standalone_freerun');exec(compile(patched,str(out/'fastexec-standalone.py'),'exec'),module.__dict__)
 fr=module.FreeRun(c,1,'cuda',compile_=False,v2=True);fr.load(0,spec.tok('GO'))
 report={'mode':'DMC port on faithful compact Triton circuit, fp32 CUDA graph; not stock HF','elf_sha256':ELF_SHA,'reference_trace_sha256':TRACE_SHA,'reference_tokens_total':TOTAL_TOKENS,'reference_tokens_used_as_inputs':False,'reference_machine_run_called':False,'torch':torch.__version__,'triton':triton.__version__,'gpu':torch.cuda.get_device_name(),'ram_base':spec.RAM_BASE,'ram_bytes':4*spec.RAM_WORDS,'drom_base':spec.DROM_BASE,'drom_bytes':4*spec.DROM_WORDS,'graph':{'chip_layers':c.n_layers(),'dims':len(c.dims)},'setup_seconds':time.perf_counter()-started,'source_adaptation':'Standalone FreeRun omits merged-chat-only Q embedding table family. Memory map uses frozen prepare.py. Executor2 arithmetic kernels unchanged.'}
 with torch.no_grad():
  t=time.perf_counter();assert fr.run(1)[0].tolist()==expected[:1].tolist();report['capture_warmup_seconds']=time.perf_counter()-t;fr.load(0,spec.tok('GO'))
 return fr,expected,spec,report

def save_checkpoint(fr,offset,path):
 import torch
 assert torch.count_nonzero(fr.ex.y).item()==0
 heads=[s for s in fr.ex.steps if s['kind']=='head']
 data={'elf_sha256':ELF_SHA,'offset':offset,'x':fr.x.detach().cpu().clone(),'tok':fr.tok.detach().cpu().clone(),'heads':[{'S':s['S'].detach().cpu().clone(),'hist':s['hist'].detach().cpu().clone()} for s in heads]}
 temporary=path.with_suffix('.tmp');torch.save(data,temporary);temporary.replace(path)

def restore_checkpoint(fr,path):
 import torch
 data=torch.load(path,map_location='cpu',weights_only=True);assert data['elf_sha256']==ELF_SHA
 heads=[s for s in fr.ex.steps if s['kind']=='head'];assert len(heads)==len(data['heads'])
 for state,saved in zip(heads,data['heads']):
  assert state['S'].shape==saved['S'].shape and state['hist'].shape==saved['hist'].shape
  state['S'].copy_(saved['S']);state['hist'].copy_(saved['hist'])
 fr.x.copy_(data['x']);fr.tok.copy_(data['tok']);fr.ex.y.zero_()
 return data['offset']
