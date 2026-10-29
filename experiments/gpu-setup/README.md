# Windows GPU environment

The GPU steps ([`memory-map-cuda`](../memory-map-cuda/README.md) and [`dmc-cuda`](../dmc-cuda/README.md)) ran on native Windows with an RTX 5070 Ti (16 GB). They don't need WSL, a system CUDA toolkit, a C compiler or pretrained model weights.

| | Tested version |
| --- | --- |
| Python | 3.14.2 |
| PyTorch | 2.14.1+cu130 (CUDA runtime 13.0) |
| Triton | `triton-windows==3.8.0.post29` |
| NVIDIA driver | 617.42 |
| Qwenputer | `ca13310e2f9f788d1dd3feeea8213c1b0c23913c` |

Official cu130 Windows wheels don't provide torch 2.15.0, so this environment uses torch 2.14.1, which satisfies Qwenputer's pinned requirement. `requirements-cuda.txt` is the full resolved package list.

## Set up

From the repository root, in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install --no-cache-dir transformers==5.21.0 numpy==2.5.4 pillow==13.0.0
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --no-deps --require-hashes -r experiments\gpu-setup\triton-requirement.txt
git clone https://github.com/divagr18/qwenputer.git .upstream\qwenputer
git -C .upstream\qwenputer checkout --detach ca13310e2f9f788d1dd3feeea8213c1b0c23913c
```

These are the commands the tested environment was built with. Qwenputer isn't installed as a package: each experiment copies the pinned checkout into a scratch directory and imports it from there.

`triton-requirement.txt` pins the Windows Triton wheel by SHA-256 `1d0fd6a6cf36ba5cced997177068e328d015e42cd4300f6995fff5a6f8707aff`. Its PyPI metadata is recorded in `package-provenance.json`. The wheel comes from the [Triton Windows fork](https://github.com/triton-lang/triton-windows/blob/readme/README.md), which supports sm120 and pairs torch 2.14 with Triton 3.8.

## The one source adaptation

The GPU runs use upstream's `tritonexec.Executor2`, driven by `fastexec.FreeRun`. It runs the circuit's sparse fp32 arithmetic in fused Triton kernels under a CUDA graph, and each output token picks the next embedding on the GPU.

Upstream `FreeRun` tries to build a token for the Q family, which exists only in the merged chat model, not in the standalone CPU. `standalone-freerun.diff` makes the copied module enumerate the standalone families plus controls instead. Q is the last family, so every other family ID and embedding row keeps its index. Circuit construction and the Triton kernels are unchanged, and the upstream checkout stays clean: the experiments apply this to a scratch copy.
