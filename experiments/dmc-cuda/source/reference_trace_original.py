#!/usr/bin/env python3
import pathlib,sys,functools,array,hashlib,json
sys.path.insert(0,'/tmp/qwenputer-dmc-port/map4')
from chipformer.rv import elf,ref
ref.decode=functools.lru_cache(None)(ref.decode)
m=ref.Machine.from_program(elf.program('/tmp/qwenputer-dmc-port/rv-build/doom.elf'));b=array.array('I');n=0;h=hashlib.sha256();chunks=[]
with open('/tmp/qwenputer-dmc-port/reference.tokens.u32','wb') as f:
 while not m.halted:
  b.extend(m.record())
  while len(b)>=65536:
   block=b[:65536];del b[:65536];raw=block.tobytes();f.write(raw);h.update(raw);n+=len(block);chunks.append({'end_token':n,'sha256':hashlib.sha256(raw).hexdigest()})
 raw=b.tobytes();f.write(raw);h.update(raw);n+=len(b);chunks.append({'end_token':n,'sha256':hashlib.sha256(raw).hexdigest()})
r={'kind':'private reference trace; not generated neural tokens','uint32_endian':sys.byteorder,'tokens':n,'sha256':h.hexdigest(),'chunks':chunks};pathlib.Path('/tmp/qwenputer-dmc-port/reference-trace.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='chunks'})
