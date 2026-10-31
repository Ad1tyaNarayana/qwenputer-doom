"""CPU checks for frame assembly, private staging and checkpoint recovery."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import showcase_worker as worker


class LocalChecks(unittest.TestCase):
    def test_column_major_and_doubled_pixels(self):
        palette = np.arange(256, dtype='<u2').tobytes()
        for flags in (0, 1):
            frame = np.zeros((240, 320), dtype='<u2')
            covered = np.zeros_like(frame, dtype=bool)
            d = dict(x=3, y=4, n=2, h=3, flags=flags)
            worker.draw_strip(frame, covered, d, palette, bytes([1, 2, 3, 4, 5, 6]))
            expected = np.array([[1, 4], [2, 5], [3, 6]])
            np.testing.assert_array_equal(frame[4:7, 3:3 + 2 * (flags + 1)], np.repeat(expected, flags + 1, axis=1))
            with self.assertRaises(AssertionError):
                worker.draw_strip(frame, covered, d, palette, bytes(6))

    def test_status_band_retained(self):
        frame = np.full((240, 320), 99, dtype='<u2')
        covered = np.zeros_like(frame, dtype=bool)
        palette = np.arange(256, dtype='<u2').tobytes()
        worker.draw_strip(frame, covered, dict(x=0, y=0, n=320, h=200, flags=0), palette, bytes([7]) * 64000)
        self.assertTrue((frame[:200] == 7).all())
        self.assertTrue((frame[200:] == 99).all())
        self.assertEqual(covered.sum(), 64000)

    def test_png_matches_raw_pixels(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            frame = np.zeros((240, 320), dtype='<u2')
            frame[0, :3] = [0xf800, 0x07e0, 0x001f]
            path = Path(tmp) / 'frame.png'
            worker.write_frame(frame, path)
            self.assertEqual(path.with_suffix('.rgb565').read_bytes(), frame.tobytes())
            self.assertEqual([tuple(v) for v in np.asarray(Image.open(path))[0, :3]], [(255, 0, 0), (0, 255, 0), (0, 0, 255)])

    def test_checkpoint_and_retained_prefix(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            packed = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
            fr = SimpleNamespace(x=torch.tensor([[4.0, 8.0]]), tok=torch.tensor([22]),
                 ex=SimpleNamespace(y=torch.zeros(1, 2), steps=[
                     {'kind': 'head', 'S': packed[:, :2], 'hist': torch.full((1, 3, 2), 3.0)},
                     {'kind': 'head', 'S': packed[:, 2:], 'hist': torch.full((1, 3, 2), 5.0)}]))
            fusion = {'scratch_y': torch.zeros(3, 2)}
            surface = np.full((240, 320), 65535, dtype='<u2')
            coverage = np.zeros_like(surface, dtype=bool)
            coverage[:3] = True
            identity = {'fixture': 'test'}
            metadata = {'identity': identity, 'offset': 2, 'chunk_seconds': 0.75, 'frames': [], 'strips': [{'serial': 1}]}
            worker.save_checkpoint(fr, fusion, root / 'checkpoint.pt', metadata, surface, coverage)
            packed.zero_()
            fr.x.zero_()
            fr.tok.zero_()
            for head in fr.ex.steps:
                head['hist'].zero_()
            restored, got_surface, got_coverage = worker.restore_checkpoint(fr, fusion, root / 'checkpoint.pt', identity)
            self.assertEqual(restored, metadata)
            torch.testing.assert_close(packed, torch.arange(16, dtype=torch.float32).reshape(1, 4, 4))
            self.assertEqual(fr.tok.item(), 22)
            self.assertEqual(fr.ex.steps[1]['hist'][0, 0, 0].item(), 5)
            np.testing.assert_array_equal(got_surface, surface)
            np.testing.assert_array_equal(got_coverage, coverage)
            with self.assertRaises(AssertionError):
                worker.restore_checkpoint(fr, fusion, root / 'checkpoint.pt', {'fixture': 'wrong'})
            trace = root / 'actual.tokens.u32'
            expected = np.array([11, 22, 33], dtype='<u4')
            trace.write_bytes(expected.tobytes())
            self.assertEqual(worker.verify_prefix(trace, expected, 2), hashlib.sha256(expected[:2].tobytes()).hexdigest())
            self.assertEqual(trace.stat().st_size, 8)
            trace.write_bytes(np.array([10, 22], dtype='<u4').tobytes())
            with self.assertRaises(AssertionError):
                worker.verify_prefix(trace, expected, 2)
            fusion['scratch_y'][0, 0] = 1
            with self.assertRaises(AssertionError):
                worker.save_checkpoint(fr, fusion, root / 'bad.pt', metadata, surface, coverage)

    def test_manifest(self):
        addresses = {'qp_strip_' + k: i * 4 for i, k in enumerate(('ptr', 'x', 'y', 'h', 'n', 'flags', 'serial'))}
        addresses['pump_lut'] = 32
        m = {'elf_sha256': 'test', 'metadata_addresses': addresses,
             'strips': [{'frame': 1, 'generated_token_offset': 3}, {'frame': 2, 'generated_token_offset': 7}],
             'frames': [{'frame': 1, 'generated_token_offset': 4}, {'frame': 2, 'generated_token_offset': 8}]}
        _, frames = worker.validate_manifest(m, {'tokens': 8}, 'walk', 2)
        self.assertEqual(frames[-1]['generated_token_offset'], 8)
        m['strips'][1]['generated_token_offset'] = 3
        with self.assertRaises(AssertionError):
            worker.validate_manifest(m, {'tokens': 8}, 'walk', 2)

    def test_private_staging_and_windows_replace_retry(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            probe, out = root / 'probe', root / 'out'
            (probe / 'helpers').mkdir(parents=True)
            (probe / 'inputs').mkdir()
            (probe / 'private-run').mkdir()
            evidence = probe / 'private-run/evidence'
            evidence.write_bytes(b'historical')
            runtime = 'from pathlib import Path\nROOT=Path(__file__).resolve().parent\ndef prepare(upstream,scratch,ram,drom):\n return str(upstream)\n'
            (probe / 'runtime.py').write_text(runtime)
            for name in ('doom.elf', 'reference.tokens.u32', 'strips.json', 'reference-trace.json'):
                (probe / 'inputs' / name).write_text('{}')
            module = worker.isolated_runtime(probe, probe / 'inputs', out, root / 'upstream')
            self.assertEqual(module.ROOT, out / 'runtime/probe')
            self.assertEqual(module.prepare('ignored', None, 64, 4096), str(root / 'upstream'))
            self.assertEqual(evidence.read_bytes(), b'historical')
            self.assertTrue(all(not p.is_symlink() for p in out.rglob('*')))
            calls = []
            class Temporary:
                def replace(self, path):
                    calls.append(path)
                    if len(calls) < 3:
                        raise PermissionError('held reader')
            with patch.object(worker.time, 'sleep'):
                worker.replace_shared(Temporary(), root / 'target')
            self.assertEqual(len(calls), 3)

    def test_prepared_walk_inputs(self):
        root = Path(__file__).resolve().parents[2] / '.cache/doom-showcase/walk-inputs'
        if not (root / 'reference-trace.json').exists():
            self.skipTest('Generate private walk inputs first')
        report = json.loads((root / 'reference-trace.json').read_text())
        manifest = json.loads((root / 'strips.json').read_text())
        markers, frames = worker.validate_manifest(manifest, report, 'walk', 3)
        self.assertEqual([f['tokens'] for f in frames], [6343110, 4456441, 4474457])
        self.assertEqual([f['generated_token_offset'] for f in frames], [6343110, 10799551, 15274008])
        self.assertEqual(worker.sha(root / 'reference.tokens.u32'), report['sha256'])
        self.assertEqual(worker.sha(root / 'doom.elf'), manifest['elf_sha256'])
        trace = np.memmap(root / 'reference.tokens.u32', mode='r', dtype='<u4')
        self.assertEqual(len(trace), 15274008)
        start = 0
        for chunk in report['chunks']:
            stop = chunk['end_token']
            self.assertEqual(stop - start, chunk['tokens'])
            self.assertEqual(hashlib.sha256(trace[start:stop].tobytes()).hexdigest(), chunk['sha256'])
            start = stop
        self.assertEqual(start, len(trace))
        self.assertEqual(len(markers), 65)
        spec = worker.load_module('local_reference_spec', root / 'reference-machine/chipformer/rv/spec.py')
        self.assertEqual({int(trace[m['generated_token_offset'] - 1]) for m in markers}, {spec.tok('BYTE', 256)})

    def test_fusion_acceptance(self):
        repo = Path(__file__).resolve().parents[2]
        hashes = worker.fusion_acceptance(repo)
        self.assertEqual(len(hashes['feedback']), 64)


if __name__ == '__main__':
    unittest.main()
