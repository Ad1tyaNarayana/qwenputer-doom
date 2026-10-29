#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import struct, pathlib, hashlib

def inspect(path):
 b=pathlib.Path(path).read_bytes();h=struct.unpack_from('<16sHHIIIIIHHHHHH',b)
 assert b[:7]==b'\x7fELF\x01\x01\x01' and h[2]==243
 ss=[struct.unpack_from('<10I',b,h[6]+i*h[11]) for i in range(h[12])]
 namesec=ss[h[13]];names=b[namesec[4]:namesec[4]+namesec[5]]
 sections=[];symbols={}
 for s in ss:
  name=names[s[0]:].split(b'\0',1)[0].decode()
  if s[2]&2:sections.append({'name':name,'address':s[3],'bytes':s[5],'write':bool(s[2]&1),'execute':bool(s[2]&4)})
  if s[1]==2:
   st=ss[s[6]];strings=b[st[4]:st[4]+st[5]]
   for off in range(s[4],s[4]+s[5],s[9]):
    ni,value,size,info,other,shndx=struct.unpack_from('<IIIBBH',b,off);n=strings[ni:].split(b'\0',1)[0].decode()
    if n and shndx:symbols[n]=value
 return {'sha256':hashlib.sha256(b).hexdigest(),'entry':h[4],'elf_flags':h[7],'sections':sections,'symbols':symbols}

if __name__=='__main__':
 import sys,json
 print(json.dumps(inspect(sys.argv[1]),indent=2))
