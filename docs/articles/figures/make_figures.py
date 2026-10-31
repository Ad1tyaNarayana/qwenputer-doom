#!/usr/bin/env python3
"""Generate article figures from published JSON, without running an experiment.

Run from any directory with the repository's .venv/bin/python. Matplotlib is
not installed there, so SVG is written directly and Pillow rasterizes the same
primitives to PNG. No measurement is embedded in this script.
"""
import json
import math
from pathlib import Path
from statistics import median
from html import escape
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#777777', '#333333']
FONT = next((p for p in [Path('/System/Library/Fonts/Supplemental/Arial.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if p.exists()), None)
assert FONT, 'Install a TrueType font or add its path to FONT.'


def read(path):
    return json.loads((ROOT / path).read_text())


class Figure:
    """One drawing instruction stream, shared by vector and raster outputs."""
    def __init__(self, title, subtitle, height=760):
        self.height = height
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">', '<rect width="100%" height="100%" fill="#fafafa"/>']
        self.im = Image.new('RGB', (1200, height), '#fafafa')
        self.draw = ImageDraw.Draw(self.im)
        self.text(45, 47, title, 28)
        self.text(45, 80, subtitle, 18, '#555555')

    def text(self, x, y, value, size=18, color='#222222', anchor='start'):
        value = str(value)
        self.svg.append(f'<text x="{x}" y="{y}" font-family="Arial, DejaVu Sans, sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>')
        self.draw.text((x, y), value, font=ImageFont.truetype(str(FONT), size), fill=color, anchor={'start':'ls', 'middle':'ms', 'end':'rs'}[anchor])

    def rect(self, x, y, w, h, color):
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}"/>')
        self.draw.rectangle((x, y, x+w, y+h), fill=color)

    def line(self, x1, y1, x2, y2, color='#cccccc', width=1):
        self.svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')
        self.draw.line((x1,y1,x2,y2), fill=color, width=width)

    def save(self, name):
        (OUT / f'{name}.svg').write_text('\n'.join(self.svg + ['</svg>']) + '\n')
        self.im.save(OUT / f'{name}.png')
        print(name)


original = read('experiments/dmc-cuda-profile/evidence/profile-summary.json')['windows']['frame2_render']
rom = read('experiments/dmc-feedback-fusion/evidence/rom-profile.json')['windows']['frame2_render']
fused = read('experiments/dmc-feedback-fusion/evidence/new-profile.json')['windows']['frame2_render']
assert original['graph_kernel_nodes_per_token'] == 133
assert rom['circuit_kernels_per_token'] == fused['circuit_kernels_per_token'] == 48
assert rom['feedback_kernels_per_token'] == 25 and fused['feedback_kernels_per_token'] == 1
# Normalize attribution names only; the original capture does not split apply.
a = original['device_groups_us_per_token']
series = [[a['data_rom_heads'], a['ram_heads'], a['code_rom_heads'], a['alu_decode_ffn'], a['other_head'], a['apply_residual_history'], a['feedback_embedding'], a['trace_device_copy'] + a['device_gaps']]]
for d in [rom, fused]:
    g = d['groups_us_per_token']
    series.append([g['data_rom_heads'], g['ram_heads'], g['code_rom_heads'], g['alu_decode_ffn'], g['other_heads'], g['data_rom_apply'] + g['other_residual_history_apply'], g['feedback_embedding'], g['trace_copy'] + g['gaps']])
spans = [original['device_stream_span_us_per_token'], rom['stream_us_per_token'], fused['stream_us_per_token']]
assert all(math.isclose(sum(v), s, abs_tol=1e-6) for v,s in zip(series,spans))
f = Figure('Where device-stream time goes', 'Frame-2 rendering windows, reference-seeded; separate profiler captures', 720)
for tick in range(0,601,100):
    x=230+tick*1.45; f.line(x,115,x,435); f.text(x,465,tick,16,anchor='middle')
for i,(label,values) in enumerate(zip(['Original','ROM fusion','ROM + feedback'],series)):
    y=145+i*105; f.text(210,y+28,label,20,anchor='end');x=230
    for value,color in zip(values,COLORS):
        f.rect(x,y,value*1.45,48,color);x+=value*1.45
    f.text(x+10,y+30,f'{sum(values):.1f}',18)
f.text(660,505,'Profiler device-stream span (µs/token)',20,anchor='middle')
for i,label in enumerate(['Data-ROM heads','RAM heads','Code-ROM heads','ALU / decode FFNs','Other heads','Residual / history apply','Feedback / embedding','Trace copy + stream gaps']):
    x=45+(i%3)*385;y=555+(i//3)*40;f.rect(x,y-16,18,18,COLORS[i]);f.text(x+28,y,label,17)
f.text(45,695,'Stream spans are not wall time. Captures use different harness revisions.',17)
f.save('profile-breakdown')

bench = read('experiments/dmc-feedback-fusion/evidence/bench.json')
summary = read('experiments/dmc-feedback-fusion/evidence/timing-summary.json')
f = Figure('Normal generation path gets faster', 'Medians and full min–max ranges; five alternating repeats of each executor', 840)
for panel,window in enumerate(['boot','frame2_render']):
    top=130+panel*335
    f.text(45,top,'Boot' if panel==0 else 'Frame-2 rendering (reference-seeded)',22)
    for j,(metric,title,maximum) in enumerate([('us','Generation time (µs/token)',600),('rate','Throughput (tokens/s)',5000)]):
        left=250+j*470;scale=390/maximum
        f.text(left+195,top+37,title,18,anchor='middle')
        for tick in range(0,maximum+1,100 if j==0 else 1000):
            x=left+tick*scale;f.line(x,top+50,x,top+245);f.text(x,top+270,tick,15,anchor='middle')
        for i,side in enumerate(['baseline','rom','fused']):
            d=bench['windows'][window][side];runs=d['runs'];assert len(runs)==5
            rates=[r['tokens_per_second'] for r in runs]
            assert all(r['match']['mismatches']==0 and r['match']['tokens']==bench['tokens'] for r in runs)
            assert math.isclose(median(rates),summary[window][side]['tokens_per_second_median'])
            vals=[r['seconds']/bench['tokens']*1e6 for r in runs] if metric=='us' else rates
            value=median(vals);lo=min(vals);hi=max(vals);y=top+65+i*64
            if j==0:f.text(left-15,y+18,['Original','ROM fusion','ROM + feedback'][i],18,anchor='end')
            f.rect(left,y,value*scale,23,COLORS[i]);f.line(left+lo*scale,y+11,left+hi*scale,y+11,'#111111',3)
            for end in [lo,hi]:f.line(left+end*scale,y+5,left+end*scale,y+17,'#111111',2)
            f.text(left,y+47,f'{value:,.2f}  [{lo:,.2f}, {hi:,.2f}]',16)
f.text(45,815,'Includes allocation, graph replay, feedback, trace copies, sync and CPU transfer. Excludes reset and checking.',16)
f.save('generation-timing')

# Complete first-frame generation sums, not wall-time speedup comparisons.
historical = read('experiments/dmc-cuda/evidence/full-report.json')
full_runs = []
for name in ['ff-full', 'unified-full']:
    report = read(f'experiments/doom-showcase/evidence/{name}-report.json')
    validation = read(f'experiments/doom-showcase/evidence/{name}-validation.json')
    assert report['complete'] and report['halted']
    assert report['mismatches'] == validation['differing_tokens'] == 0
    assert report['generated_tokens'] == validation['tokens_compared'] == historical['generated_tokens']
    assert report['interruptions'] == report['started_from_checkpoint_offset'] == 0
    assert report['uninterrupted_wall_claim_valid']
    assert math.isclose(report['retained_chunk_seconds'], validation['retained_chunk_seconds'])
    assert math.isclose(report['uninterrupted_end_to_end_seconds'], validation['uninterrupted_end_to_end_seconds'])
    assert report['inputs']['doom.elf'] == historical['elf_sha256']
    assert report['actual_trace_sha256'] == validation['actual_trace_sha256'] == historical['actual_trace_sha256']
    full_runs.append(report)
values = [historical['compute_seconds_total']] + [r['retained_chunk_seconds'] for r in full_runs]
original_layers = historical['graph']['chip_layers']
assert full_runs[0]['runtime']['graph']['chip_layers'] == original_layers
unified_layers = full_runs[1]['runtime']['geometry']['layers']
labels = [f'Historical: original ({original_layers} layers)',
          f'Run 1: original ({original_layers} layers)',
          f'Run 2: unified W=4 ({unified_layers} layers)']
f = Figure('Complete first-frame generation', 'Same ELF and trace; run 2 evaluates a different network', 600)
left = 380
axis_max = math.ceil(max(values) / 500) * 500
scale = 680 / axis_max
for tick in range(0, axis_max + 1, 500):
    x = left + tick * scale
    f.line(x, 125, x, 415)
    f.text(x, 450, f'{tick:,}', 16, anchor='middle')
for i, (label, value) in enumerate(zip(labels, values)):
    y = 150 + i * 95
    f.text(left - 18, y + 28, label, 18, anchor='end')
    f.rect(left, y, value * scale, 42, COLORS[i])
    f.text(left + value * scale + 12, y + 28, f'{value:,.2f} s', 18)
f.text(720, 495, 'Generation time (seconds)', 20, anchor='middle')
f.text(45, 545, 'Successful generation chunk sums. Historical run resumed; both new runs were uninterrupted.', 17)
f.text(45, 575, 'Unified depth excludes stock padding. Run 2 is not a same-network executor speedup comparison.', 17)
f.save('full-frame-comparison')

counts = read('experiments/one-instr-design/counts-walk.json')
ops=sorted(counts['by_op'].items(),key=lambda item:-item[1]['tokens'])
assert sum(v['tokens'] for _,v in ops)==counts['tokens']
rows=[(k,v['tokens']) for k,v in ops[:7]]+[('Other operations + refresh',sum(v['tokens'] for _,v in ops[7:]))]
f=Figure('Which instructions produce the trace?', 'Ten-frame continuing walk, including startup and tail; software-reference counts',760)
axis_max = math.ceil(max(value / counts['tokens'] * 100 for _, value in rows) / 5) * 5
share_scale = 650 / axis_max
for tick in range(0, axis_max + 1, 5):
    x=330+tick*share_scale;f.line(x,115,x,625);f.text(x,655,tick,17,anchor='middle')
for i,(name,value) in enumerate(rows):
    y=130+i*60;f.text(310,y+23,name,18,anchor='end');f.rect(330,y,value/counts['tokens']*100*share_scale,32,COLORS[i]);f.text(340+value/counts['tokens']*100*share_scale,y+23,f'{value:,}',17)
f.text(710,695,'Share of all trace tokens (%)',20,anchor='middle')
f.text(45,735,f'Total: {counts["tokens"]:,} tokens. This is instruction mix, not per-frame phase attribution.',17)
f.save('continuing-instruction-mix')

model=read('experiments/one-instr-design/cost-model.json')
assert math.isclose(model['baseline_us_per_token'],summary['frame2_render']['fused']['total_us_per_token'],abs_tol=.001)
values=[model['old_tokens_per_frame']*summary['frame2_render']['baseline']['total_us_per_token']/1e6,model['old_tokens_per_frame']*summary['frame2_render']['fused']['total_us_per_token']/1e6,model['scenarios'][0]['seconds_per_frame']]
f=Figure('Projected continuing-frame time remains large', 'Count-derived projections, not measured neural frame times',620)
left=330;span=760
for power in range(-2,5):
    x=left+(power+2)/6*span;f.line(x,115,x,440);f.text(x,475,f'{10**power:g}',17,anchor='middle')
for i,(label,value) in enumerate(zip(['Original executor','ROM + feedback','Hypothetical 1 instruction/step'],values)):
    y=175+i*105;x=left+(math.log10(value)+2)/6*span
    f.text(310,y,label,18,anchor='end');f.rect(x-7,y-15,14,20,COLORS[i]);f.text(x+18,y,f'{value:,.1f} s/frame',19)
f.text(710,525,'Projected generation time per frame (seconds, logarithmic axis)',20,anchor='middle')
f.text(45,570,'Counts average later frames in the short walk. The hypothetical design assumes unchanged step cost.',17)
f.text(45,600,'The target budget and gap ratios are explained in the article; no target timing is measured here.',17)
f.save('frame-time-gap')
