# Expanded memory on the GPU

The actual 64 KiB RAM / 4 MiB DROM circuit generated 142 exact tokens on the RTX 5070 Ti using upstream Triton Executor2. The test printed `ABC` and halted. All 65,536 bytes extracted directly from GPU recurrent state matched the reference RAM and the independently verified Mac proof. This is a memory test, not Doom execution.

The map has RAM at `0x01000000`, DROM at `0x02000000`, and unchanged 256 KiB code ROM at `0x00040000`. The circuit has 44 chip layers, 627 key heads, and 1,247 value heads. The workload covers initial snapshot sentinels, first/middle/last banks, signed and unsigned widths, merged stores, and control flow. It is the successful main case from the CPU suite, not all twelve fault cases.

The one short run took 4.62 seconds for setup, 0.633 seconds for graph capture/warm-up, 0.0682 seconds for generation, and 0.00403 seconds for extraction. Generation includes self-fed graph execution and CPU token transfer, excluding setup, capture, reference generation, and extraction. Peak PyTorch CUDA allocation was 278,903,808 bytes. These are short correctness-probe measurements, not sustained Doom throughput.

`probe.py` and `helpers/` are the executed sources. `artifacts/` contains the actual trace, assembly, input sentinels, memory patch manifest, extracted and expected RAM, report, and independent validation. The unchanged decoder reads only executor state and head metadata. Reference tokens never feed generation. The sole FreeRun adaptation omits the merged-chat-only Q family while preserving standalone IDs.

First set up the [Windows GPU environment](../gpu-setup/README.md). Copy this experiment to a fresh scratch location to preserve the archived evidence:

```powershell
Copy-Item -Recurse experiments\memory-map-cuda .cache\memory-map-cuda-run
.\.venv\Scripts\python.exe .cache\memory-map-cuda-run\probe.py --upstream .upstream\qwenputer
```

Use a fresh destination each time. The probe rejects an existing scratch package. It enforces a 290-second watchdog and an 8 GiB CUDA allocator cap. No pretrained weights are used.
