"""Export hashes and shapes only, leaving actual private state and traces in place."""
import hashlib,json,platform,sys
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
out=ROOT/'private-run'
report=json.loads((out/'full-report.json').read_text());assert report.get('full_frame_completed') and 'error' not in report
proof={'collected_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),'packages':{name:version(name) for name in ['torch','triton-windows','numpy','pillow']},'cuda_runtime':torch.version.cuda,'strip_hashes':[]}
for path in sorted((out/'gpu-strips').glob('*.npz')):
 with np.load(path,allow_pickle=False) as data:
  descriptor=json.loads(str(data['descriptor']));indices=data['indices'].tobytes();palette=data['palette'].tobytes()
 values=np.frombuffer(palette,dtype='<u2')[np.frombuffer(indices,dtype='u1')].reshape(descriptor['n'],descriptor['h']).T
 if descriptor['flags']&1:values=np.repeat(values,2,axis=1)
 proof['strip_hashes'].append({'descriptor':descriptor,'indices_bytes':len(indices),'indices_sha256':hashlib.sha256(indices).hexdigest(),'palette_sha256':hashlib.sha256(palette).hexdigest(),'rgb565_sha256':hashlib.sha256(values.astype('<u2').tobytes()).hexdigest()})
assert len(proof['strip_hashes'])==25
checkpoint=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True)
assert checkpoint['offset']==report['generated_tokens']
proof['checkpoint']={'offset':checkpoint['offset'],'elf_sha256':checkpoint['elf_sha256'],'file_bytes':(out/'checkpoint.pt').stat().st_size,'sha256':hashlib.sha256((out/'checkpoint.pt').read_bytes()).hexdigest(),'next_embedding_shape':list(checkpoint['x'].shape),'last_token_shape':list(checkpoint['tok'].shape),'head_tensors':[{'S_shape':list(h['S'].shape),'hist_shape':list(h['hist'].shape),'dtype':str(h['S'].dtype)} for h in checkpoint['heads']]}
proof['actual_artifacts']={name:{'bytes':(out/name).stat().st_size,'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest()} for name in ['actual.tokens.u32','frame.rgb565','frame.png','final-gpu-ram.bin']}
(out/'final-proof-metadata.json').write_text(json.dumps(proof,indent=2));print(json.dumps({'strips':len(proof['strip_hashes']),'checkpoint_offset':proof['checkpoint']['offset'],'actual_artifacts':proof['actual_artifacts']},indent=2))
