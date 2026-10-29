#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Give the upstream converter ELF-style action names on Mach-O hosts."""
import os
import re
import subprocess
import sys

result = subprocess.run([os.environ.get('DMC_REAL_NM', 'nm'), *sys.argv[1:]],
                        capture_output=True, text=True)
sys.stdout.write(re.sub(r'(?<!\w)_A_', 'A_', result.stdout))
sys.stderr.write(result.stderr)
raise SystemExit(result.returncode)
