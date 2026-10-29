#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import argparse,importlib.util,pathlib,sys,json,time,struct,hashlib
from elf_info import inspect
p=argparse.ArgumentParser();p.add_argument('--prepare',type=pathlib.Path,required=True);p.add_argument('--upstream',type=pathlib.Path,required=True);p.add_argument('--scratch',type=pathlib.Path,required=True);p.add_argument('--elf',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--max-instructions',type=int,default=20000000);a=p.parse_args()
spec=importlib.util.spec_from_file_location('prepare_map',a.prepare);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);manifest=mod.prepare(a.upstream,a.scratch,8192,8192)
sys.path.insert(0,str(a.scratch));from chipformer.rv import elf,ref,spec as S
import functools
ref.decode=functools.lru_cache(maxsize=None)(ref.decode)
m=ref.Machine.from_program(elf.program(a.elf));t=time.time();tokens=0;last=0
a.out.mkdir(parents=True,exist_ok=True)
while not m.halted and m.steps<a.max_instructions:
 tokens+=len(m.record())
 if m.steps-last>=1000000:
  last=m.steps; print(m.steps,hex(m.pc),bytes(m.out[-200:]),flush=True)
result={'steps':m.steps,'tokens':tokens,'halted':m.halted,'exit_code':m.exit_code,'fault':m.fault,'pc':m.pc,'elapsed_seconds':time.time()-t,'output':m.out.decode(errors='replace'),'map_manifest':manifest}
info=inspect(a.elf); result['elf']={k:v for k,v in info.items() if k!='symbols'}
def word(name):
 addr=info['symbols'].get(name)
 return m.ram[(addr-S.RAM_BASE)//4] if addr is not None else None
result['state']={n:word(n) for n in ['gamestate','gameepisode','gamemap','gametic','leveltime','qwenputer_frame_ready','qwenputer_framebuffer','qwenputer_frame_bytes']}
result['heap']={'start':info['symbols']['__heap_start'],'end':info['symbols']['__heap_end'],'high_water':word('heap'),'used_bytes':word('heap')-info['symbols']['__heap_start']}
if result['state']['qwenputer_frame_ready']:
 ptr=result['state']['qwenputer_framebuffer']; size=result['state']['qwenputer_frame_bytes']
 raw=struct.pack('<%dI'%len(m.ram),*m.ram)[ptr-S.RAM_BASE:ptr-S.RAM_BASE+size]
 (a.out/'frame.raw').write_bytes(raw)
 result['frame']={'width':320,'height':200,'format':'little-endian XRGB8888 (B,G,R,unused bytes)','sha256':hashlib.sha256(raw).hexdigest(),'unique_pixels':len(set(struct.unpack('<64000I',raw)))}
 from PIL import Image
 Image.frombytes('RGB',(320,200),raw,'raw','BGRX').save(a.out/'frame.png')
(a.out/'report.json').write_text(json.dumps(result,indent=2)+'\n');(a.out/'output.txt').write_bytes(m.out)
(a.out/'ram.bin').write_bytes(struct.pack('<%dI'%len(m.ram),*m.ram))
print(json.dumps({k:v for k,v in result.items() if k!='map_manifest'},indent=2))

if not (m.halted and m.fault is None and m.exit_code == 0 and result['state']['qwenputer_frame_ready'] and result['state']['qwenputer_frame_bytes'] == 320 * 200 * 4):
 sys.exit('Doom first-frame gate failed; inspect report.json')
