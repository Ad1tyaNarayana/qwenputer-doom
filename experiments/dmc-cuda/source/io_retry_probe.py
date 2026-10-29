import json,threading,time
from runtime import ROOT
from full_worker import atomic_json
path=ROOT/'private-run/io-boundary-proof.json';path.write_text('{"old":true}')
held=path.open('rb');timer=threading.Timer(0.2,held.close);timer.start();started=time.perf_counter();atomic_json(path,{'new':True});elapsed=time.perf_counter()-started;timer.join();assert json.loads(path.read_text())=={'new':True} and elapsed>=0.15
result={'held_reader_released_after_seconds':0.2,'replacement_succeeded':True,'elapsed_seconds':elapsed,'changes':'External file replacement only; no circuit or game state changes'}
(ROOT/'private-run/io-retry-proof.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
