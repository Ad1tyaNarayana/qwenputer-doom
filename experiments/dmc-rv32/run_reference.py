# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
import pathlib,sys,importlib.util,functools,struct,json,time,hashlib,argparse
from elf_info import inspect
p=argparse.ArgumentParser();p.add_argument('--prepare',type=pathlib.Path,required=True);p.add_argument('--upstream',type=pathlib.Path,required=True);p.add_argument('--elf',type=pathlib.Path,required=True);p.add_argument('--scratch',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--max-instructions',type=int,default=30000000);a=p.parse_args()
prep=a.prepare;sp=importlib.util.spec_from_file_location('prep',prep);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod);mod.prepare(a.upstream,a.scratch,64,4096)
sys.path.insert(0,str(a.scratch));from chipformer.rv import elf,ref,spec as S
ref.decode=functools.lru_cache(None)(ref.decode);path=a.elf;info=inspect(path);syms=info['symbols'];m=ref.Machine.from_program(elf.program(path));tokens=0;t=time.time();last=0;strips=[];pixels=[0]*(320*240);written=set(); min_sp=S.SP_INIT
def word(name):return m.ram[(syms[name]-S.RAM_BASE)//4]
def byte(addr):
 assert S.RAM_BASE<=addr<S.RAM_BASE+4*S.RAM_WORDS,hex(addr)
 return (m.ram[(addr-S.RAM_BASE)//4]>>((addr&3)*8))&255
def strip():
 d={k:word('qp_strip_'+k) for k in ['ptr','x','y','h','n','flags','serial']};assert d['x']<320 and d['y']+d['h']<=240,d
 mul=2 if d['flags']&1 else 1;assert d['x']+d['n']*mul<=320,d
 for c in range(d['n']):
  for y in range(d['h']):
   idx=byte(d['ptr']+c*d['h']+y);addr=syms['pump_lut']+2*idx;rgb=byte(addr)|(byte(addr+1)<<8)
   for k in range(mul):pos=(d['y']+y)*320+d['x']+c*mul+k;pixels[pos]=rgb;written.add(pos)
 d['generated_token_offset']=tokens;d['instruction_offset']=m.steps;strips.append(d)
while not m.halted and m.steps<a.max_instructions:
 rec=m.record();tokens+=len(rec);min_sp=min(min_sp,m.x[2])
 if rec==[S.tok('BYTE',256)]:strip()
 if m.steps-last>=1000000:last=m.steps;print(m.steps,hex(m.pc),'strips',len(strips),bytes(m.out[-150:]),flush=True)
a.out.mkdir(parents=True,exist_ok=True);raw=struct.pack('<76800H',*pixels);(a.out/'frame.rgb565').write_bytes(raw)
from PIL import Image
rgb=bytes(v for c in pixels for v in [((c>>11)&31)*255//31,((c>>5)&63)*255//63,(c&31)*255//31]);Image.frombytes('RGB',(320,240),rgb).save(a.out/'frame.png')
report={'kind':'DMC E1M1 software reference execution; no compiled neural execution','instructions':m.steps,'tokens':tokens,'halted':m.halted,'exit_code':m.exit_code,'fault':m.fault,'pc':m.pc,'elapsed_seconds':time.time()-t,'output':m.out.decode(errors='replace'),'memory':{'frame_base':word('qp_frame_base'),'frame_bytes':word('qp_frame_bytes'),'plan_arena_bytes':word('qp_plan_arena'),'arena_end':syms['__ram_end'],'stack_top':S.SP_INIT,'minimum_sp':min_sp,'stack_used':S.SP_INIT-min_sp},'strips':len(strips),'written_pixels':len(written),'frame_sha256':hashlib.sha256(raw).hexdigest(),'unique_colors':len(set(pixels)),'game_state':{n:word(n) for n in ['gamestate','gameepisode','gamemap','gametic','leveltime'] if n in syms},'elf':{k:v for k,v in info.items() if k!='symbols'}}
(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(a.out/'strips.json').write_text(json.dumps({'kind':'Reference-derived stop offsets only; no pixels or model state','token_offsets':'Cumulative record tokens; excludes prefill and GO','elf_sha256':info['sha256'],'metadata_addresses':{n:v for n,v in syms.items() if n.startswith('qp_strip_') or n=='pump_lut'},'strips':strips},indent=2)+'\n');(a.out/'ram.bin').write_bytes(struct.pack('<%dI'%len(m.ram),*m.ram));print(json.dumps(report,indent=2))

expected = (m.halted and m.exit_code == 0 and m.fault is None
            and b'QWENPUTER_DMC_FRAME_READY\n' in m.out
            and report['strips'] > 0 and report['written_pixels'] == 320 * 240
            and report['unique_colors'] > 1
            and report['game_state'].get('gamestate') == 0
            and report['game_state'].get('gameepisode') == 1
            and report['game_state'].get('gamemap') == 1
            and report['memory']['frame_base'] + report['memory']['frame_bytes']
                <= report['memory']['arena_end'] - report['memory']['plan_arena_bytes']
            and report['memory']['minimum_sp'] >= report['memory']['arena_end'])
if not expected:
    raise SystemExit('DMC first-frame gate failed; inspect report.json')
