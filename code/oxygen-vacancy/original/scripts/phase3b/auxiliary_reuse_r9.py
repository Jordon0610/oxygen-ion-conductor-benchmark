"""Reuse authenticated identical DFT selection; refresh altered queries by final allowed-source refit."""
from common import *
from sklearn.model_selection import GroupKFold
HIST=ROOT.parents[1]/'results/Route_A_Publication_Consolidated_v1/02_COMPLETE_PREVIOUS_MASTER/02_research_repository'
def auxiliary_job(job):
 path=OUT/'auxiliary_cache'/(job['job_id']+'.json')
 if path.exists():return path.name+' completed'
 p=plan();ids=job['allowed_formula_ids'];assert ids==sorted(ids)
 oldpath=HIST/'results/phase3b/auxiliary_cache'/(job['job_id']+'.json');old=json.loads(oldpath.read_text());hp=json.loads((HIST/'results/phase3b/analysis_plan_phase3b.json').read_text())
 assert old['plan_sha256']==sha(HIST/'results/phase3b/analysis_plan_phase3b.json')
 for field in ['allowed_formula_ids','arm','model','target','seed','source_id']:assert old[field]==job[field],field
 for relative in ['results/phase3a_1/composition_features.csv','results/phase3a_1/source_clean_host_targets.csv']:
  assert sha(ROOT/relative)==sha(HIST/relative)==hp['source_hashes'][relative],relative
 assert p['auxiliary_policy']['candidates']==hp['auxiliary_policy']['candidates'] and p['auxiliary_policy']['fixed']==hp['auxiliary_policy']['fixed']
 X=pd.read_csv(ROOT/'results/phase3a_1/composition_features.csv',index_col='formula_id',float_precision='round_trip').loc[ids]
 H=pd.read_csv(ROOT/'results/phase3a_1/source_clean_host_targets.csv',float_precision='round_trip');H=H[H.population.eq('B')].set_index('formula_id').loc[ids];y=H[job['target']].copy();y.name=job['target']
 if job['arm'].startswith('shuffle_'):y[:]=np.random.default_rng(job['seed']).permutation(y.to_numpy())
 elif job['arm'].startswith('random_'):y[:]=np.random.default_rng(job['seed']).standard_normal(len(y))
 assert digest(y)==old['generated_target_hash']
 groups=pd.factorize(H.chemical_system,sort=True)[0];inner=[]
 for i,(a,b) in enumerate(GroupKFold(5,shuffle=True,random_state=314159).split(X,groups=groups)):inner.append({'fold':i,'train_ids':[ids[j] for j in a],'validation_ids':[ids[j] for j in b]})
 assert inner==old['inner_splits'];candidates=p['auxiliary_policy']['candidates'][job['model']];assert candidates[int(np.argmin(old['candidate_scores']))]==old['params']
 for relative in ['scripts/phase3b/common.py','scripts/phase3b/auxiliary.py']:
  assert sha(ROOT/relative)==sha(HIST/relative),relative
 E=pd.read_csv(ROOT/'results/phase3a_1/experimental_composition_features.csv',index_col='row_id',float_precision='round_trip').loc[job['prediction_row_ids']]
 OE=pd.read_csv(HIST/'results/phase3a_1/experimental_composition_features.csv',index_col='row_id',float_precision='round_trip');assert list(E.columns)==list(OE.columns)
 unchanged=[r for r in E.index if r in old['prediction_row_ids'] and np.array_equal(E.loc[r].to_numpy(),OE.loc[r].to_numpy(),equal_nan=True)]
 refit=len(unchanged)!=len(E);z=pd.Series(old['predictions'],index=old['prediction_row_ids']);traces=old['traces']
 if refit:
  pred,trace=fit_predict(job['model'],old['params'],X,y,E,'R9_allowed_population_refit_same_historical_selection',dft=True)
  traces=[trace];delta=float(np.max(np.abs(pred[[E.index.get_loc(r) for r in unchanged]]-z.loc[unchanged].to_numpy()))) if unchanged else 0.;assert delta<=1e-6,delta
 else:pred=z.loc[E.index].to_numpy();delta=0.
 evidence={'historical_cache_path':str(oldpath),'historical_cache_SHA256':sha(oldpath),'historical_plan_SHA256':old['plan_sha256'],'DFT_training_X_hash':digest(X),'DFT_training_y_hash':digest(y),'inner_memberships_identical':True,'fixed_parameters_grid_seed_implementation_environment_identical':True,'selection_reused':'authenticated unchanged DFT-only candidate scores and selected parameters','final_DFT_fit_executed':refit,'unchanged_query_count':len(unchanged),'max_unchanged_query_difference_eV':delta,'R0023_query_changed':'R0023' in E.index and 'R0023' not in unchanged,'external_Ea_access':False}
 js(path,dict(**job,params=old['params'],candidate_scores=old['candidate_scores'],inner_splits=inner,predictions=pred.tolist(),generated_target_hash=digest(y),traces=traces,selection_traces_source=old['traces'],dependency_reuse_evidence=evidence,plan_sha256=sha(OUT/'analysis_plan_phase3b.json')))
 return path.name
