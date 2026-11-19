"""DFT-only auxiliary training. This module never opens experimental outcomes."""
from common import *
from sklearn.model_selection import GroupKFold

def auxiliary_job(job):
 path=OUT/'auxiliary_cache'/(job['job_id']+'.json')
 if path.exists():return path.name+' cached'
 p=plan();ids=job['allowed_formula_ids'];assert ids==sorted(ids)
 X=pd.read_csv(ROOT/'results/phase3a_1/composition_features.csv',index_col='formula_id',float_precision='round_trip').loc[ids]
 H=pd.read_csv(ROOT/'results/phase3a_1/source_clean_host_targets.csv',float_precision='round_trip');H=H[H.population.eq('B')].set_index('formula_id').loc[ids]
 y=H[job['target']].copy();y.name=job['target']
 if job['arm'].startswith('shuffle_'):y[:]=np.random.default_rng(job['seed']).permutation(y.to_numpy())
 elif job['arm'].startswith('random_'):y[:]=np.random.default_rng(job['seed']).standard_normal(len(y))
 groups=pd.factorize(H.chemical_system,sort=True)[0];splits=[]
 for i,(a,b) in enumerate(GroupKFold(5,shuffle=True,random_state=314159).split(X,groups=groups)):
  tr=[ids[j] for j in a];va=[ids[j] for j in b];assert set(H.loc[tr,'chemical_system']).isdisjoint(H.loc[va,'chemical_system']);splits.append(dict(fold=i,train_ids=tr,validation_ids=va))
 candidates=p['auxiliary_policy']['candidates'][job['model']];scores=[];traces=[]
 for ci,params in enumerate(candidates):
  losses=[]
  for s in splits:
   tr=s['train_ids'];va=s['validation_ids'];pred,trace=fit_predict(job['model'],params,X.loc[tr],y.loc[tr],X.loc[va],f'candidate{ci}/inner{s["fold"]}',dft=True)
   losses.extend(np.abs(y.loc[va].to_numpy()-pred));traces.append(trace)
  scores.append(float(np.mean(losses)))
 best=candidates[int(np.argmin(scores))]
 E=pd.read_csv(ROOT/'results/phase3a_1/experimental_composition_features.csv',index_col='row_id',float_precision='round_trip').loc[job['prediction_row_ids']]
 pred,trace=fit_predict(job['model'],best,X,y,E,'allowed_population_refit',dft=True);traces.append(trace)
 js(path,dict(**job,params=best,candidate_scores=scores,inner_splits=splits,predictions=pred.tolist(),generated_target_hash=digest(y),traces=traces,plan_sha256=sha(OUT/'analysis_plan_phase3b.json')))
 return path.name
