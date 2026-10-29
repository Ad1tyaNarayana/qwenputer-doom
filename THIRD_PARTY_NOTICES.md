# Qwenputer

The runtime imports Qwenputer from https://github.com/divagr18/qwenputer at revision ca13310e2f9f788d1dd3feeea8213c1b0c23913c. The memory-map and GPU experiments adapt upstream chipformer/rv/tiny.py and its executors.

MIT License

Copyright (c) 2026 Divyansh Agrawal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
# Doom ports

`experiments/dmc-rv32/platform.c` derives from DMC100-Doom's platform stub and retains its GPL-2.0-or-later notice. DMC100-Doom is by Artur Bieniek, derived from id Software's Doom and Doomgeneric. The source used here is the Aspire PIXO port in atc1441/Vape_DOOM_ScreenShare, pinned in that experiment's README. Its COPYING and NOTICE accompany the adaptation. The other new glue files carry their licenses individually. A combined Doom executable remains subject to the engine's GPL terms.

`experiments/doom-baremetal` includes Doom's GPL text and Zig compiler_rt's MIT notice for reproducing the linked runtime. No upstream engine binary, WAD, converted game-data pack, or WAD-bearing ELF is included. See the experiments' NOTICE files for corresponding-source and attribution requirements.
