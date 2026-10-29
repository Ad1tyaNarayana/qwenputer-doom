#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import argparse, pathlib, re, subprocess, json
p=argparse.ArgumentParser(); p.add_argument('--zig',required=True); p.add_argument('--doomgeneric',required=True,type=pathlib.Path); p.add_argument('--wad',type=pathlib.Path); p.add_argument('--out',type=pathlib.Path,required=True); a=p.parse_args()
r=pathlib.Path(__file__).resolve().parent; s=a.doomgeneric.resolve()/'doomgeneric'; out=a.out.resolve(); out.mkdir(parents=True,exist_ok=True)
assert subprocess.check_output(['git','-C',str(s.parent),'rev-parse','HEAD'],text=True).strip()=='e257a09527062bfb986b6e5e0780839b6cbd5b71'
if subprocess.check_output(['git','-C',str(s.parent),'status','--porcelain'],text=True).strip():
 raise RuntimeError('Doomgeneric checkout must be clean for a reproducible build')
cc=[str(pathlib.Path(a.zig).resolve()),'cc','-target','riscv32-linux-musl','-mcpu=generic_rv32+m+zba','-mabi=ilp32','-Os','-g','-fno-pic','-fno-builtin','-ffunction-sections','-fdata-sections','-DNORMALUNIX','-DLINUX','-DSNDSERV','-D_DEFAULT_SOURCE','-DDOOMGENERIC_RESX=320','-DDOOMGENERIC_RESY=200','-Wno-implicit-function-declaration','-I'+str(s)]
names=re.search(r'^SRC_DOOM = (.+)$',(s/'Makefile').read_text(),re.M)[1].split(); names.remove('doomgeneric_xlib.o')
commands=[]; objects=[]
with (out/'build.log').open('w') as log:
 for name in names+['platform.o','minilib.o','libc_extra.o','start.o','wad.o']:
  obj=out/name; src=s/name.replace('.o','.c') if name in names else r/name.replace('.o','.c')
  if name=='start.o': src=r/'start.S'
  if name=='wad.o':
   src=out/'wad.S'; src.write_text('.section .wad,"a",@progbits\n'+('.incbin "'+str(a.wad.resolve())+'"\n' if a.wad else ''))
  cmd=cc+['-c',str(src),'-o',str(obj)]; commands.append(cmd); subprocess.run(cmd,check=True,stdout=log,stderr=log); objects.append(str(obj))
 rt=out/'compiler_rt.a'; rtcmd=[cc[0],'build-lib',str(pathlib.Path(cc[0]).parent/'lib/compiler_rt.zig'),'-target','riscv32-freestanding','-mcpu','generic_rv32+m+zba','-O','ReleaseSmall','-ffunction-sections','-fdata-sections','-femit-bin='+str(rt)]; commands.append(rtcmd); subprocess.run(rtcmd,check=True,stdout=log,stderr=log)
 cmd=cc+['-static','-nostdlib','-fuse-ld=lld']+objects+[str(rt),'-Wl,--gc-sections','-Wl,-T,'+str(r/'port.ld'),'-o',str(out/'doom.elf')]; commands.append(cmd); subprocess.run(cmd,check=True,stdout=log,stderr=log)
(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n'); print(out/'doom.elf')
