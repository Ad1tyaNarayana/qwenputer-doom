# SPDX-License-Identifier: MIT
"""Write a private, little-endian RV reference trace for comparison only."""
import argparse,array,functools,hashlib,json,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--elf',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--max-instructions',type=int,default=10000000);args=p.parse_args()
 if args.trace.exists() or args.report.exists():raise ValueError('Use fresh private output files.')
 sys.path.insert(0,str(args.package.resolve()))
 from chipformer.rv import elf,ref,spec
 assert spec.RAM_BASE==0x01000000 and spec.RAM_WORDS==16384 and spec.DROM_BASE==0x02000000 and spec.DROM_WORDS==1048576
 ref.decode=functools.lru_cache(None)(ref.decode)
 machine=ref.Machine.from_program(elf.program(args.elf));buffer=array.array('I');assert buffer.itemsize==4
 count=0;digest=hashlib.sha256();chunks=[]
 def write(block,output):
  nonlocal count
  if sys.byteorder!='little':block.byteswap()
  raw=block.tobytes();output.write(raw);digest.update(raw);count+=len(block);chunks.append({'end_token':count,'sha256':hashlib.sha256(raw).hexdigest()})
 args.trace.parent.mkdir(parents=True,exist_ok=True);args.report.parent.mkdir(parents=True,exist_ok=True)
 with args.trace.open('wb') as output:
  while not machine.halted and machine.steps<args.max_instructions:
   buffer.extend(machine.record())
   while len(buffer)>=65536:
    block=buffer[:65536];del buffer[:65536];write(block,output)
  if buffer:write(buffer,output)
 assert machine.halted and machine.fault is None and machine.exit_code==0
 report={'kind':'private reference trace; not generated neural tokens','uint32_endian':'little','tokens':count,'sha256':digest.hexdigest(),'chunks':chunks}
 args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='chunks'}))

if __name__=='__main__':main()
