#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare private E1M1 tables and NOR data without changing the pinned source."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

REVISION = '710f452b59decb07e8b34bad9a66875657d63484'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True, help='Clean PixoAspire_DOOM_Port source')
p.add_argument('--wad', type=Path, required=True, help='Explicit local IWAD; never downloaded')
p.add_argument('--out', type=Path, required=True, help='Fresh private ignored output directory')
p.add_argument('--nm', type=Path, help='Native llvm-nm or nm executable')
p.add_argument('--dwarfdump', type=Path, help='Native llvm-dwarfdump executable')
a = p.parse_args()
source, out, wad = (v.resolve() for v in (a.source, a.out, a.wad))
head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
if head != REVISION:
    raise ValueError(f'Expected {REVISION}, got {head}')
if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain',
                            '--untracked-files=all'], text=True):
    raise ValueError('Use a clean pinned source checkout. Build only in its private copy.')
if not wad.is_file():
    raise ValueError(f'Local IWAD missing: {wad}')
if out.exists() and any(out.iterdir()):
    raise ValueError('Choose a fresh empty private output directory.')


def native_tool(explicit, names, apple_name=None):
    if explicit:
        tool = str(explicit.resolve())
    else:
        tool = next((v for n in names if (v := shutil.which(n))), None)
        if tool is None and sys.platform == 'darwin' and apple_name:
            result = subprocess.run(['xcrun', '--find', apple_name], capture_output=True, text=True)
            if result.returncode == 0:
                tool = result.stdout.strip()
    if not tool or not Path(tool).is_file():
        raise ValueError('Tool unavailable: ' + '/'.join(names))
    return tool


nm = native_tool(a.nm, ['llvm-nm', 'nm'])
dwarf = native_tool(a.dwarfdump, ['llvm-dwarfdump'], 'llvm-dwarfdump')
out.mkdir(parents=True, exist_ok=True)
(out / '.gitignore').write_text('*\n')
copy = out / 'source'
shutil.copytree(source, copy, ignore=shutil.ignore_patterns('build', 'gen', '__pycache__', '*.pyc'))
# Sparse source checkouts may omit the one upstream image-building helper.
# Recover that exact tracked file from the already-verified local Git revision.
mknor = copy / 'tests/host/mknor.py'
if not mknor.exists():
    root = Path(subprocess.check_output(['git', '-C', str(source), 'rev-parse',
                                         '--show-toplevel'], text=True).strip())
    relative = (source.relative_to(root) / 'tests/host/mknor.py').as_posix()
    content = subprocess.check_output(['git', '-C', str(source), 'show', 'HEAD:' + relative])
    mknor.parent.mkdir(parents=True, exist_ok=True)
    mknor.write_bytes(content)
# The upstream fetch verifies its own pinned Doomgeneric source. It writes
# only inside our private source copy, never to the supplied checkout.
env = os.environ.copy()
env['NM'] = str(Path(__file__).resolve().parent / 'nm_action_names.py')
env['DMC_REAL_NM'] = nm
env['DWARFDUMP'] = dwarf
with (out / 'prepare.log').open('w') as log:
    subprocess.run(['sh', 'oracle/fetch.sh'], cwd=copy, env=env, check=True,
                   stdout=log, stderr=log)
    subprocess.run([sys.executable, '-I', 'tools/conv/__main__.py', '--wad', str(wad),
                    '--gen', str(out / 'gen'), '--out', str(out / 'packs'),
                    '--packs', 'E1M1', 'all'], cwd=copy, env=env, check=True,
                   stdout=log, stderr=log)
    subprocess.run([sys.executable, str(copy / 'tests/host/mknor.py'),
                    str(out / 'packs/E1M1'), str(out / 'nor.img')], cwd=copy,
                   env=env, check=True, stdout=log, stderr=log)
print(f'Generated tables: {out / "gen"}\nPrivate NOR image: {out / "nor.img"}')
