import hashlib,json,os,threading,time
from pathlib import Path
from runtime import ROOT,build,save_checkpoint,restore_checkpoint

def main():
 timer=threading.Timer(120,lambda:os._exit(124));timer.daemon=True;timer.start()
 out=ROOT/'private-run';out.mkdir(exist_ok=True);report={'phase':'prefix','wall_limit_seconds':120}
 try:
  import numpy as np,torch
  fr,expected,spec,report_base=build();report.update(report_base)
  offset=0;compute_seconds=0;started=time.perf_counter();trace_hash=hashlib.sha256()
  with (out/'actual.tokens.u32').open('wb') as trace,torch.no_grad():
   while offset<65536:
    amount=min(8192,65536-offset);t=time.perf_counter();generated=fr.run(amount)[0];compute_seconds+=time.perf_counter()-t
    wanted=expected[offset:offset+amount]
    if not np.array_equal(generated,wanted):
     index=int(np.flatnonzero(generated!=wanted)[0]);raise AssertionError({'position':offset+index,'actual':int(generated[index]),'expected':int(wanted[index])})
    raw=generated.astype('<u4').tobytes();trace.write(raw);trace.flush();trace_hash.update(raw);offset+=amount
    progress={'generated_tokens':offset,'mismatches':0,'compute_seconds':compute_seconds,'elapsed_seconds_with_comparison_io':time.perf_counter()-started,'actual_trace_sha256':trace_hash.hexdigest()};(out/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True)
  report.update(progress);report['generation_seconds_sum_chunks_one_run']=compute_seconds
  checkpoint=out/'checkpoint.pt';save_checkpoint(fr,offset,checkpoint)
  fr.load(0,spec.tok('GO'));assert restore_checkpoint(fr,checkpoint)==offset
  with torch.no_grad():resumed=fr.run(256)[0]
  assert np.array_equal(resumed,expected[offset:offset+256])
  (out/'resume-proof.tokens.u32').write_bytes(resumed.astype('<u4').tobytes())
  report.update({'checkpoint_offset':offset,'checkpoint_restore_verified_future_tokens':256,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'timed_work':'Sum of8 FreeRun CUDA graph chunks with self-fed embeddings, output copies and CPU transfer; excludes setup, graph capture, reference comparison, file IO and checkpoint save/restore','full_frame_completed':False,'gpu_strips_extracted':0,'private_checkpoint_bytes':checkpoint.stat().st_size})
 except Exception as e:
  import traceback
  report['error']=repr(e);report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
 finally:
  (out/'prefix-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True);timer.cancel()
 if 'error' in report:raise SystemExit(1)

if __name__=='__main__':main()
