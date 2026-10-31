"""Render the accepted normal-path timing comparison as a standalone chart."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
report = json.loads((root / 'evidence/checks.json').read_text())
assert report['accepted']
sections = []
labels = [('baseline','Original',1),('rom','ROM fusion',2),('fused','ROM + feedback',3)]
for window, title in [('boot','Boot'),('frame2_render','Rendering, reference-seeded')]:
    w = report['timing'][window]
    maximum = w['baseline']['total_us_per_token']
    rows = []
    for side, name, color in labels:
        value = w[side]['total_us_per_token']
        rows.append(f'<div class="row"><span>{name}</span><div class="track"><div class="bar" style="width:{100*value/maximum:.3f}%;background:var(--chart-{color})"></div></div><strong>{value:.2f}</strong></div>')
    sections.append(f'<section><h3>{title}</h3>'+''.join(rows)+f'<p>{w["cumulative_speedup"]:.3f}× vs original · {w["incremental_speedup"]:.3f}× vs ROM fusion</p></section>')
html = '''<!doctype html><html><head><meta charset="utf-8"><style>
body{font-family:var(--font-sans);color:var(--foreground);margin:0} .grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}h3{font-size:14px;font-weight:600;margin:12px 0 16px}.row{display:grid;grid-template-columns:105px 1fr 48px;gap:8px;align-items:center;font-size:12px;margin:12px 0}.track{height:13px;background:var(--secondary);border-radius:3px;overflow:hidden}.bar{height:13px;border-radius:3px}strong{font-family:var(--font-mono);font-weight:500;font-size:11px;text-align:right}p{font-size:12px;color:var(--muted-foreground);margin:14px 0}.caption{font-size:12px;margin:0 0 8px;color:var(--muted-foreground)}@media(max-width:580px){.grid{grid-template-columns:1fr;gap:8px}}
</style></head><body><p class="caption">Normal-path µs/token · median of five alternating repeats · lower is faster</p><div class="grid">'''+''.join(sections)+'''</div><p class="caption">65,536 tokens per call. Exact state gates passed in both windows. This is not a full-frame timing.</p></body></html>'''
(root / 'timing.html').write_text(html)
print(root / 'timing.html')
