from common import *
from concurrent.futures import ProcessPoolExecutor,as_completed
from downstream import baseline_job,augmented_job
from auxiliary import auxiliary_job
import time

def execute(fn,jobs,label):
 start=time.time();print(label,'jobs',len(jobs),flush=True)
 with ProcessPoolExecutor(max_workers=6) as pool:
  futures=[pool.submit(fn,j) for j in jobs]
  for i,f in enumerate(as_completed(futures),1):
   result=f.result()
   if i%10==0 or i==len(jobs):print(label,i,'/',len(jobs),'elapsed_s',round(time.time()-start),result,flush=True)
 js(OUT/(label+'_completed.json'),dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs=len(jobs),seconds=time.time()-start))

def run():
 p=plan()
 for f,h in p['source_hashes'].items():assert sha(ROOT/f)==h,f
 for filename,h in [('frozen_eligible_membership.csv',p['frozen_membership']['sha256']),('experimental_configurations.json',p['frozen_membership']['configurations_sha256']),('auxiliary_jobs.json',p['auxiliary_policy']['jobs_sha256'])]:assert sha(OUT/filename)==h,filename
 if not (OUT/'fit_started.json').exists():js(OUT/'fit_started.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(OUT/'analysis_plan_phase3b.json')))
 configs=read('experimental_configurations.json');tasks=[(c,m) for c in configs for m in c['downstream_models']]
 execute(baseline_job,tasks,'baseline_selection')
 execute(auxiliary_job,read('auxiliary_jobs.json'),'auxiliary_fitting')
 execute(augmented_job,tasks,'matched_downstream_fitting')
 js(OUT/'fit_completed.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(OUT/'analysis_plan_phase3b.json'),outer_configurations=len(configs)))
 print('All fitting completed; no test outcomes scored during fitting.',flush=True)
if __name__=='__main__':run()
