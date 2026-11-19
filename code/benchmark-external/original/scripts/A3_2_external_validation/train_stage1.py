"""Stage 1. This process never opens an external label-bearing CSV."""
from pathlib import Path
import sys,json,hashlib,datetime,os,warnings
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,joblib
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from benchmark.engine import tune_and_fit
from benchmark.design import MODELS
from benchmark.features import load,featurizers,one_record
from A2_applicability_domain.domain import fit_geometry,chemistry
OUT=ROOT/'results/A3_2_external_validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,o):
 with (OUT/n).open('x') as f:json.dump(o,f,indent=2,ensure_ascii=False);f.write('\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 assert not (OUT/'final_model_manifest.json').exists() and not (OUT/'stage2_started.json').exists()
 assert sha(OUT/'analysis_plan_A3_2.json')==(OUT/'analysis_plan_A3_2.sha256').read_text().strip()
 plan=json.loads((OUT/'analysis_plan_A3_2.json').read_text());meta=pd.read_csv(ROOT/'results/benchmark/cohort_manifest.csv');k=meta[meta.cohort_K].set_index('row_id')
 assert k.index.tolist()==plan['internal_training']['row_ids']
 X=load().loc[k.index];y=k.Ea.astype(float);splits=json.loads((OUT/'internal_cv_membership.json').read_text())
 # This allowlisted file physically contains no external numerical activation energy.
 ext=pd.read_csv(OUT/'label_free_external_metadata.csv').set_index('external_record_id')
 mag,sto=featurizers()
 ex=pd.DataFrame([one_record({'normalization_status':r.normalization_status,'Formula':r.numeric_formula,'final_component':r.conductivity_component.replace('verified_','')},mag,sto) for _,r in ext.iterrows()],index=ext.index,columns=X.columns).replace([np.inf,-np.inf],np.nan)
 ex.to_csv(OUT/'external_features_without_targets.csv',float_format='%.17g')
 # Verify deterministic featurization against the actual frozen internal feature table.
 regenerated=np.array([one_record(r,mag,sto) for r in k.reset_index().to_dict('records')]);assert np.allclose(regenerated,X.to_numpy(),rtol=1e-12,atol=1e-10,equal_nan=True)
 (OUT/'models').mkdir(exist_ok=True);manifest={};candidates=[];traces=[]
 def trace(e):
  traces.append(e)
  with (OUT/'internal_fit_trace.jsonl').open('a') as f:f.write(json.dumps(e)+'\n')
 with threadpool_limits(limits=2):
  for name in MODELS:
   print('Internal-only model selection:',name,flush=True)
   model,best,cand=tune_and_fit(name,X,y,splits,'A3_2_internal_final',trace)
   file=OUT/'models'/(name+'.joblib');joblib.dump(model,file,compress=3)
   manifest[name]={'status':'success','parameters':best,'file':str(file.relative_to(ROOT)),'sha256':sha(file),'training_rows':312,'fit_ids':k.index.tolist()}
   candidates.extend([{'model':name,**r} for r in cand])
   print('Frozen final fit:',name,best,flush=True)
  # The domain fit receives composition X only, and an internal sentinel to keep API unchanged.
  compcols=[c for c in X if not c.startswith('component_')];assert len(compcols)==139
  _,art=fit_geometry(X[compcols].to_numpy(),X[compcols].iloc[:1].to_numpy())
  distances=cdist(art['train_PCs'],art['train_PCs']);g=k.connected_group_id.to_numpy();distances[g[:,None]==g[None,:]]=np.inf
  nearest=distances.argmin(1);loo=distances.min(1);assert (g!=g[nearest]).all();q=float(np.quantile(loo,.95,method='linear'));assert q>0
  np.savez_compressed(OUT/'internal_domain_transform.npz',**art,group_LOO=loo,q95_group=np.array(q),group_LOO_neighbor=nearest)
  pd.DataFrame({'row_id':k.index,'connected_group_id':g,'group_LOO_distance':loo,'nearest_other_group_row_id':k.index[nearest],'nearest_other_group_id':g[nearest]}).to_csv(OUT/'domain_reference_training_rows.csv',index=False)
  chemistry_internal={r:chemistry(f) for r,f in k.normalized_formula.items()}
  save('domain_reference_manifest.json',{'training_ids':k.index.tolist(),'composition_columns':compcols,'q95_group':q,'retained_features':int(art['retained_feature_mask'].sum()),'retained_PCs':len(art['PCA_components']),'explained_fraction':float(art['explained_variance_ratio'].sum()),'elements':sorted({e for a in chemistry_internal.values() for e in a[0]}),'chemical_systems':sorted({a[1] for a in chemistry_internal.values()}),'fit_inputs':'internal composition descriptors only; no outcome input'})
 pd.DataFrame(candidates).to_csv(OUT/'internal_cv_candidate_results.csv',index=False)
 for t in traces:
  assert set(t['fit_ids'])<=set(k.index)
  if t['model']!='DummyRegressor':assert t['preprocessing_fit_ids']==t['fit_ids']
 files=['external_features_without_targets.csv','internal_cv_membership.json','internal_cv_candidate_results.csv','internal_domain_transform.npz','domain_reference_manifest.json','domain_reference_training_rows.csv','internal_fit_trace.jsonl']
 save('stage1_checks.json',{'all_fits_internal_only':True,'external_labels_loaded':False,'frozen_internal_features_reproduced':True,'all_144_feature_names_unchanged':True,'CV_group_disjoint':all(set(k.loc[s['train_ids']].connected_group_id).isdisjoint(k.loc[s['validation_ids']].connected_group_id) for s in splits),'training_group_reference_neighbors_differ':True,'model_training_attempts':len(traces),'external_metadata_source':'label_free_external_metadata.csv only'})
 save('final_model_manifest.json',{'frozen_at_utc':now(),'stage':'1_complete_FE_internal_only_selection_external_labels_previously_exposed','plan_sha256':sha(OUT/'analysis_plan_A3_2.json'),'models':manifest,'frozen_artifact_hashes':{f:sha(OUT/f) for f in files},'stage1_script_sha256':sha(Path(__file__)),'external_target_access_count':0,'no_fitting_permitted_after_this_manifest':True,'software':{n:__import__(n).__version__ for n in ['numpy','pandas','sklearn','xgboost','joblib']}})
 (OUT/'final_model_manifest.sha256').write_text(sha(OUT/'final_model_manifest.json')+'\n')
 print('FINAL MODEL MANIFEST FROZEN. External Ea has not been loaded.',flush=True)
if __name__=='__main__':main()
