#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Link the pinned DMC Doom fork to Qwenputer, with local generated data."""
import argparse
import json
from pathlib import Path
import subprocess
from elf_info import inspect
from workload import configure

REVISION = '710f452b59decb07e8b34bad9a66875657d63484'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--zig', type=Path, required=True)
p.add_argument('--source', type=Path, required=True, help='PixoAspire_DOOM_Port directory')
p.add_argument('--gen', type=Path, required=True, help='Private converter generated tables')
p.add_argument('--nor', type=Path, required=True, help='Private E1M1 NOR image from mknor.py')
p.add_argument('--libc', type=Path, required=True, help='Sibling doom-baremetal glue directory')
p.add_argument('--out', type=Path, required=True)
p.add_argument('--frames', type=int, help='Opt-in continuing mode, 1 to 1000000 displayed frames')
p.add_argument('--input', type=Path, help='JSON per-tic key states; requires --frames')
a = p.parse_args()
r, gen, libc, out = (v.resolve() for v in (a.source, a.gen, a.libc, a.out))
here = Path(__file__).resolve().parent
revision = subprocess.check_output(['git', '-C', str(r), 'rev-parse', 'HEAD'], text=True).strip()
if revision != REVISION:
    raise ValueError(f'Expected source revision {REVISION}, got {revision}')
if subprocess.check_output(['git', '-C', str(r), 'status', '--porcelain', '--untracked-files=all'], text=True):
    raise ValueError('Use a clean pinned source checkout.')
out.mkdir(parents=True, exist_ok=True)
if a.input and a.frames is None:
    p.error('--input requires --frames')
if a.frames is not None:
    configure(a.frames, a.input, out)
cc = [str(a.zig.resolve()), 'cc', '-target', 'riscv32-linux-musl',
      '-mcpu=generic_rv32+m+zba', '-mabi=ilp32', '-Oz', '-g',
      '-ffunction-sections', '-fdata-sections', '-ffreestanding', '-fno-builtin',
      '-fno-common', '-fwrapv', '-fsigned-char', '-std=gnu11', '-w']
includes = ['engine/stub/include', 'engine', 'port/data', 'port/render',
            'port/obj', 'port/math', 'platform', 'port/game', 'port/ovl']
cc += ['-I' + str(r / v) for v in includes] + ['-I' + str(gen)]
sources = [v for v in (r / 'engine').glob('*.c')
           if v.name not in ('w_wad.c', 'w_file.c', 'z_zone.c')]
for directory in ['port/data', 'port/render', 'port/obj']:
    sources += list((r / directory).glob('*.c'))
sources += [r / 'port/math/trig.c', here / 'platform.c', here / 'pump.c']
sources += [gen / n for n in ['info.c', 'colormap.c', 'trig_tables.c', 'view.c']]
sources += [libc / 'minilib.c', libc / 'libc_extra.c', here / 'start.S']
# The unused FAT volume tail is zero-filled by upstream mknor.py. Its first
# E1M1 image fits below 2.5 MiB; reads beyond the retained payload return 0xff.
image = a.nor.read_bytes()
if len(image) != 4 * 1024 * 1024:
    raise ValueError('Expected the upstream 4 MiB NOR image')
length = (len(image.rstrip(b'\0')) + 4095) // 4096 * 4096
nor = out / 'nor-private.bin'
nor.write_bytes(image[:length])
assembly = out / 'nor.S'
assembly.write_text('.section .nor,"a",@progbits\n.incbin ' + json.dumps(str(nor)) + '\n')
sources.append(assembly)
commands, objects = [], []
with (out / 'build.log').open('w') as log:
    for i, source in enumerate(sources):
        obj = out / f'{i}_{source.stem}.o'
        flags = [v for v in cc if 'engine/stub/include' not in v] if source.name in (
            'minilib.c', 'libc_extra.c', 'platform.c') else cc
        if a.frames is not None and source == here / 'platform.c':
            flags = flags + ['-DQP_CONTINUING', '-I' + str(out)]
        command = flags + ['-c', str(source), '-o', str(obj)]
        commands.append(command)
        subprocess.run(command, check=True, stdout=log, stderr=log)
        objects.append(str(obj))
    runtime = out / 'compiler_rt.a'
    command = [cc[0], 'build-lib', str(Path(cc[0]).parent / 'lib/compiler_rt.zig'),
               '-target', 'riscv32-freestanding', '-mcpu', 'generic_rv32+m+zba',
               '-O', 'ReleaseSmall', '-ffunction-sections', '-fdata-sections',
               '-femit-bin=' + str(runtime)]
    commands.append(command)
    subprocess.run(command, check=True, stdout=log, stderr=log)
    command = cc + ['-nostdlib', '-static', '-fuse-ld=lld', '-Wl,--gc-sections',
                    '-Wl,-T,' + str(here / 'dmc.ld'), *objects, str(runtime),
                    '-o', str(out / 'doom.elf')]
    if a.frames is not None:
        command += ['-Wl,--wrap=TryRunTics']
    commands.append(command)
    subprocess.run(command, check=True, stdout=log, stderr=log)
(out / 'commands-private.json').write_text(json.dumps(commands, indent=2) + '\n')
info = inspect(out / 'doom.elf')
if a.frames is not None:
    workload_path = out / 'workload.json'
    workload = json.loads(workload_path.read_text())
    workload['elf_sha256'] = info['sha256']
    workload_path.write_text(json.dumps(workload, indent=2) + '\n')
info.pop('symbols')
(out / 'elf-info.json').write_text(json.dumps(info, indent=2) + '\n')
print(out / 'doom.elf')
