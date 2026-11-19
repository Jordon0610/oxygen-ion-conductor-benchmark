"""Baseline-only experimental selection; matched-arm refits accept training outcomes only."""
from common import *
import csv

def training_targets(ids):
 wanted=set(ids);values={}
 with (ROOT/'results/benchmark/cohort_manifest.csv').open(newline='') as f:
  for row in csv.DictReader(f):
   if row['row_id'] in wanted:values[row['row_id']]=float(row['Ea'])
 return pd.Series([values[i] for i in ids],index=ids,name='Ea_meV')

def tune_baseline(X,y,inner,name,candidates):
 assert X.shape[1]==144 and list(X.index)==list(y.index)
 traces=[];scores=[]
 for ci,params in enumerate(candidates):
  errors=[]
  for s in inner:
   tr=s['train_ids'];va=s['validation_ids'];assert set(tr).isdisjoint(va);assert set(tr)|set(va)==set(X.index)
   pred,trace=fit_predict(name,params,X.loc[tr],y.loc[tr],X.loc[va],f'baseline_candidate{ci}/inner{s["inner_fold"]}')
   traces.append(trace);errors.extend(np.abs(y.loc[va].to_numpy()-pred))
  scores.append(float(np.mean(errors)))
 selected=int(np.argmin(scores));return candidates[selected],scores,traces

def baseline_job(task):
 cfg,name=task;path=OUT/'baseline_cache'/(cfg['config_id']+'__'+name+'.json')
 if path.exists():return str(path.name)+' cached'
 p=plan();X=pd.read_csv(ROOT/'results/benchmark/composition_features.csv',index_col='row_id',float_precision='round_trip');tr=cfg['train_ids'];te=cfg['test_ids'];y=training_targets(tr)
 best,scores,traces=tune_baseline(X.loc[tr],y,cfg['inner_splits'],name,p['downstream_policy']['candidates'][name])
 pred,trace=fit_predict(name,best,X.loc[tr],y,X.loc[te],'baseline_outer_refit');traces.append(trace)
 js(path,dict(config_id=cfg['config_id'],model=name,params=best,candidate_scores=scores,train_ids=tr,test_ids=te,predictions=pred.tolist(),traces=traces,selection_features='baseline_only_144',plan_sha256=sha(OUT/'analysis_plan_phase3b.json')))
 return path.name

def augmented_job(task):
 cfg,name=task;path=OUT/'downstream_cache'/(cfg['config_id']+'__'+name+'.json')
 if path.exists():return str(path.name)+' cached'
 p=plan();base=json.loads((OUT/'baseline_cache'/(cfg['config_id']+'__'+name+'.json')).read_text());best=base['params']
 X=pd.read_csv(ROOT/'results/benchmark/composition_features.csv',index_col='row_id',float_precision='round_trip');tr=cfg['train_ids'];te=cfg['test_ids'];y=training_targets(tr)
 rows=[];traces=[]
 for arm in cfg['auxiliary_arms']:
  if arm=='real_Efmin_XGBaux' and name!='ExtraTreesRegressor':continue
  aux=json.loads((OUT/'auxiliary_cache'/(cfg['source_id']+'__'+arm+'.json')).read_text());assert aux['allowed_formula_ids']==cfg['allowed_DFT_formula_ids']
  z=pd.Series(aux['predictions'],index=aux['prediction_row_ids']);A=X.loc[tr].copy();V=X.loc[te].copy();A['auxiliary_scalar']=z.loc[tr];V['auxiliary_scalar']=z.loc[te]
  pred,trace=fit_predict(name,best,A,y,V,arm+'/outer_refit');trace['arm']=arm;trace['auxiliary_job_id']=aux['job_id'];traces.append(trace)
  rows.append(dict(arm=arm,predictions=pred.tolist()))
 js(path,dict(config_id=cfg['config_id'],model=name,params=best,train_ids=tr,test_ids=te,arms=rows,traces=traces,hyperparameters_source='baseline_only',plan_sha256=sha(OUT/'analysis_plan_phase3b.json')))
 return path.name
