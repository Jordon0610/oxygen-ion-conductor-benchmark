"""Scoped existing-recipe FE outer fits. Frozen memberships/grids; no outer-label tuning."""
from pathlib import Path
import sys,json,hashlib,time,os
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from benchmark.engine import tune_and_fit,fit_on_training,TrainingOnlyPreprocessor,digest
OUT=ROOT/'results/benchmark'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def verify_lock():
 assert sha(ROOT/'R9_ANALYSIS_SPEC.json')==(ROOT/'R9_ANALYSIS_SPEC.sha256').read_text().strip()
 spec=json.loads((ROOT/'R9_ANALYSIS_SPEC.json').read_text())
 for n,h in spec['locked_artifact_hashes'].items():assert sha(ROOT/n)==h,n
 return spec
def worker(job):
 verify_lock();start=time.time();sid=job['split_id'];name=job['model'];path=OUT/'checkpoints'/(job['job_id']+'.json')
 if path.exists():
  r=json.loads(path.read_text());assert r['R9_plan_sha256']==sha(ROOT/'R9_ANALYSIS_SPEC.json') and r['fit_signature']==job['fit_signature'];return {'job_id':job['job_id'],'action':r['action'],'status':'CACHED_COMPLETED','seconds':0}
 X=pd.read_csv(OUT/'composition_features.csv',index_col='row_id',float_precision='round_trip');meta=pd.read_csv(OUT/'cohort_manifest.csv').set_index('row_id')
 s=next(s for s in json.loads((OUT/'splits.json').read_text()) if s['split_id']==sid);tr=s['train_ids'];te=s['test_ids'];y=meta.loc[tr,'Ea'].astype(float)
 assert digest(X.loc[tr])==job['signature_inputs']['X_digest'] and digest(y)==job['signature_inputs']['y_digest']
 traces=[]
 def trace(e):traces.append(e)
 original=json.loads(Path(job['historical_checkpoint_PRIVATE']).read_text());assert sha(job['historical_checkpoint_PRIVATE'])==job['historical_checkpoint_sha256']
 with threadpool_limits(limits=2):
  if job['planned_action'] in ['REUSE_IDENTICAL_DEPENDENCIES','REAGGREGATE_CHANGED_EVALUATION']:
   assert s['inner_splits']==next(z for z in json.loads((ROOT/'support/historical_benchmark/splits.json').read_text()) if z['split_id']==s['historical_split_id'])['inner_splits']
   lookup=dict(zip(original['test_ids'],original['predictions']));pred=np.array([lookup[r] for r in te]);best=original['selected_parameters'];cand=original['inner_search']
   r=dict(original);r.update(test_ids=te,predictions=pred.tolist())
   action=job['planned_action'];fit_count=0
  else:
   if job['planned_action']=='REGENERATE_QUERY_ONLY':
    best=original['selected_parameters'];cand=original['inner_search']
    model=fit_on_training(name,best,X.loc[tr],y,{'split_id':sid,'stage':'outer_refit_query_only_missing_historical_binary','candidate':-1,'inner_fold':-1,'expected_ids':tr},trace)
    action='REGENERATE_QUERY_ONLY__AUTHENTICATED_SELECTION_REUSED_NEW_OUTER_FIT';fit_count=1
   else:
    model,best,cand=tune_and_fit(name,X.loc[tr],y,s['inner_splits'],sid,trace)
    action='REFIT_CHANGED_TRAINING_OR_SELECTION';fit_count=len(traces)
   pred=model.predict(X.loc[te]);assert np.isfinite(pred).all(),'Retain adverse finite outcomes; nonfinite numerical failure must be reported'
   r={'selected_parameters':best,'inner_search':cand,'test_ids':te,'predictions':pred.tolist(),'status':'success'}
   if job['planned_action']=='REGENERATE_QUERY_ONLY':
    lookup=dict(zip(original['test_ids'],original['predictions']));unchanged=[(i,rid) for i,rid in enumerate(te) if rid!='R0023']
    dif=max((abs(float(pred[i])-lookup[rid]) for i,rid in unchanged),default=0.)
    assert dif<=1e-6,('Identical-recipe query-only replay mismatch',sid,name,dif)
    r['unchanged_query_max_abs_difference_meV']=dif;r['locked_absolute_tolerance_meV']=1e-6
 r.update(split_id=sid,historical_split_id=s['historical_split_id'],model=name,R9_plan_sha256=sha(ROOT/'R9_ANALYSIS_SPEC.json'),fit_signature=job['fit_signature'],action=action,train_rows=len(tr),train_groups=meta.loc[tr,'connected_group_id'].nunique(),purged_rows=len(s['purged_ids']),elapsed_seconds=time.time()-start,fit_count_this_revision=fit_count,version='GLOBAL_FE_R9',status='success',dependency_evidence=job['signature_inputs'])
 r['train_groups']=int(r['train_groups'])
 tracefile=OUT/'fit_traces'/(job['job_id']+'.jsonl');tracefile.parent.mkdir(exist_ok=True)
 tracefile.write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in traces))
 if fit_count==0:r['reuse_receipt']={'historical_checkpoint_sha256':job['historical_checkpoint_sha256'],'ordered_training_X_y_and_inner_membership_identical':True,'implementation_identical':True,'pinned_environment_identical':True,'query_rows_identical':True,'new_fit':False}
 save(path,r)
 return {'job_id':job['job_id'],'action':action,'status':'COMPLETED','fit_count':fit_count,'seconds':round(time.time()-start,3)}
def main():
 verify_lock();jobs=json.loads((ROOT/'jobs/benchmark_jobs.json').read_text());completed=[];start=time.time()
 # Worker count is orchestration only; estimator seeds/capacities/thread counts are unchanged.
 with ProcessPoolExecutor(max_workers=6) as ex:
  futs={ex.submit(worker,j):j for j in jobs}
  for i,f in enumerate(as_completed(futs),1):
   job=futs[f]
   try:r=f.result();completed.append(r)
   except Exception as e:
    save(ROOT/'jobs/benchmark_failure.json',{'job_id':job['job_id'],'error':repr(e),'unfinished_job_ids':[j['job_id'] for j in jobs if j['job_id'] not in {x['job_id'] for x in completed}],'scope':'Affected module not certified; checkpointed completed fits retained'});raise
   if i%20==0 or i==len(jobs):
    save(ROOT/'jobs/benchmark_status.json',{'completed':completed,'requested':len(jobs),'elapsed_seconds':time.time()-start});print(f'Benchmark {i}/{len(jobs)} completed; elapsed {time.time()-start:.1f}s; {r["job_id"]}',flush=True)
 verify_lock();save(ROOT/'jobs/benchmark_fitting_completed.json',{'jobs':len(jobs),'elapsed_seconds':time.time()-start,'no_outer_labels_used_for_selection':True,'R9_plan_sha256':sha(ROOT/'R9_ANALYSIS_SPEC.json')})
if __name__=='__main__':main()
