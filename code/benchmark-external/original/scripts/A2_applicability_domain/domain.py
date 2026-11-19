from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from pymatgen.core import Composition
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/A2_applicability_domain'
MODELS={'ExtraTreesRegressor':'ExtraTrees','XGBoost':'XGBoost','RBF-SVR':'SVR','Ridge':'Ridge'}

def json_save(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def get_plan():
 assert sha(OUT/'analysis_plan_A2.json')==(OUT/'analysis_plan_A2.sha256').read_text().strip()
 plan=json.loads((OUT/'analysis_plan_A2.json').read_text())
 for relative,expected in plan['source_hashes'].items():
  assert sha(ROOT/relative)==expected, 'Frozen source changed: '+relative
 return plan

def fit_geometry(train_X,test_X):
 """The complete fitted domain procedure. Signature accepts descriptors only."""
 train=np.asarray(train_X,float).copy();test=np.asarray(test_X,float).copy()
 train[~np.isfinite(train)]=np.nan;test[~np.isfinite(test)]=np.nan
 med=np.array([np.nanmedian(v) if np.isfinite(v).any() else 0. for v in train.T])
 train=np.where(np.isnan(train),med,train);test=np.where(np.isnan(test),med,test)
 mean=train.mean(0);sd=train.std(0,ddof=0);keep=sd>.001*np.maximum(1,np.abs(mean));assert keep.any(),'No usable training features'
 tr=(train[:,keep]-mean[keep])/sd[keep];te=(test[:,keep]-mean[keep])/sd[keep]
 pca=PCA(svd_solver='full',whiten=False).fit(tr);cumulative=np.cumsum(pca.explained_variance_ratio_);needed=int(np.searchsorted(cumulative,.90)+1);count=min(20,needed)
 trpc=pca.transform(tr)[:,:count];tepc=pca.transform(te)[:,:count]
 result={};artifacts={'imputation_median':med,'training_mean':mean,'training_std':sd,'retained_feature_mask':keep,'PCA_components':pca.components_[:count],'PCA_mean':pca.mean_,'explained_variance_ratio':pca.explained_variance_ratio_[:count],'PCs_required_for_90pct':np.array(needed),'train_scaled':tr,'train_PCs':trpc}
 for name,a,b in [('pca',trpc,tepc),('raw',tr,te)]:
  loo_matrix=cdist(a,a,metric='euclidean');np.fill_diagonal(loo_matrix,np.inf);loo=loo_matrix.min(1);loo_neighbor=loo_matrix.argmin(1)
  distances=cdist(b,a,metric='euclidean');nearest=distances.argmin(1);d=distances[np.arange(len(b)),nearest];q=float(np.quantile(loo,.95,method='linear'))
  r=d/q if q>0 else np.where(d==0,0.,np.inf)
  result[name]={'d':d,'q95':q,'r':r,'neighbor':nearest,'LOO':loo,'LOO_neighbor':loo_neighbor}
 return result,artifacts

def chemistry(formula):
 elements=sorted(e.symbol for e in Composition(formula).elements)
 return elements,'-'.join(e for e in elements if e!='O')

def main():
 p=get_plan();(OUT/'fold_preprocessing').mkdir(exist_ok=True)
 # Read metadata only; Ea and predictions are not loaded until geometry artifacts are saved.
 meta_cols=['row_id','Formula','normalized_formula','normalized_doi','family','final_component','connected_group_id','cohort_K']
 meta=pd.read_csv(ROOT/'results/benchmark/cohort_manifest.csv',usecols=meta_cols);meta=meta[meta.cohort_K].set_index('row_id')
 features=pd.read_csv(ROOT/'results/benchmark/composition_features.csv',index_col=0)[p['feature_definition']['names']]
 splits=json.loads((OUT/'frozen_memberships.json').read_text());records=[];fit_rows=[];loo_rows=[];checks=[]
 chemical={r:chemistry(v) for r,v in meta.normalized_formula.items()}
 with threadpool_limits(limits=1):
  for s in splits:
   train=s['train_ids'];test=s['test_ids'];assert set(train).isdisjoint(test)
   result,art=fit_geometry(features.loc[train].to_numpy(),features.loc[test].to_numpy());np.savez_compressed(OUT/'fold_preprocessing'/f"{s['split_id']}.npz",**art)
   elements=set(e for r in train for e in chemical[r][0]);systems={chemical[r][1] for r in train}
   fit_rows.append({'split_id':s['split_id'],'protocol':s['protocol'],'n_train':len(train),'n_test':len(test),'original_features':139,'retained_features':int(art['retained_feature_mask'].sum()),'retained_PCs':len(art['PCA_components']),'PCs_required_for_90pct':int(art['PCs_required_for_90pct']),'explained_fraction':float(art['explained_variance_ratio'].sum()),'q95_PCA':result['pca']['q95'],'q95_raw':result['raw']['q95'],'train_elements':';'.join(sorted(elements)),'train_systems':';'.join(sorted(systems)),'training_ids_SHA256':hashlib.sha256(json.dumps(train).encode()).hexdigest()})
   for i,r in enumerate(train):loo_rows.append({'split_id':s['split_id'],'row_id':r,'PCA_LOO_distance':result['pca']['LOO'][i],'PCA_LOO_neighbor_row_id':train[result['pca']['LOO_neighbor'][i]],'raw_LOO_distance':result['raw']['LOO'][i],'raw_LOO_neighbor_row_id':train[result['raw']['LOO_neighbor'][i]]})
   for i,r in enumerate(test):
    m=meta.loc[r];new=sorted(set(chemical[r][0])-elements);geom=bool(result['pca']['r'][i]<=1)
    records.append({'split_id':s['split_id'],'protocol':s['protocol'],'repeat':s['repeat'],'fold':s['fold'],'row_id':r,'connected_group_id':m.connected_group_id,'formula':m.Formula,'normalized_formula':m.normalized_formula,'family':m.family,'conductivity_component':m.final_component,'chemical_system':chemical[r][1],'d_test':result['pca']['d'][i],'q95':result['pca']['q95'],'r':result['pca']['r'][i],'nearest_training_row_id':train[result['pca']['neighbor'][i]],'geometric_in_domain':geom,'unseen_element':bool(new),'unseen_elements':';'.join(new),'primary_in_domain':geom and not new,'unseen_chemical_system':chemical[r][1] not in systems,'raw_d_test':result['raw']['d'][i],'raw_q95':result['raw']['q95'],'raw_r':result['raw']['r'][i],'raw_geometric_in_domain':bool(result['raw']['r'][i]<=1),'raw_primary_in_domain':bool(result['raw']['r'][i]<=1) and not new})
   # Training-only invariance test: arbitrarily change every test descriptor.
   altered,newart=fit_geometry(features.loc[train].to_numpy(),features.loc[test].to_numpy()+1000000)
   invariance=all(np.array_equal(art[k],newart[k]) for k in art) and all(result[k]['q95']==altered[k]['q95'] and np.array_equal(result[k]['LOO'],altered[k]['LOO']) for k in result)
   assert invariance
   checks.append({'split_id':s['split_id'],'test_feature_perturbation_leaves_fitted_artifacts_and_thresholds_unchanged':invariance,'train_ids':train,'test_ids':test})
 geometry=pd.DataFrame(records);geometry.to_csv(OUT/'domain_geometry_without_outcomes.csv',index=False);pd.DataFrame(fit_rows).to_csv(OUT/'fold_preprocessing_audit.csv',index=False);pd.DataFrame(loo_rows).to_csv(OUT/'training_LOO_distances.csv',index=False);json_save(OUT/'training_only_geometry_tests.json',checks)
 # Only now load frozen prediction outcomes; no estimator fitting follows.
 pred=pd.read_csv(ROOT/'results/benchmark/outer_predictions.csv');pred=pred[(pred.cohort=='K_FE')&pred.protocol.isin(['P3','P4'])&pred.model.isin(MODELS)]
 wide=geometry.copy();true=pred[['split_id','row_id','y_true']].drop_duplicates();assert not true.duplicated(['split_id','row_id']).any()
 wide=wide.merge(true.rename(columns={'y_true':'observed_Ea'}),on=['split_id','row_id'],validate='one_to_one')
 for model,name in MODELS.items():
  q=pred[pred.model==model][['split_id','row_id','y_pred','ae','se']].rename(columns={'y_pred':name+'_prediction','ae':name+'_absolute_error','se':name+'_squared_error'})
  wide=wide.merge(q,on=['split_id','row_id'],validate='one_to_one');assert np.isfinite(wide[[name+'_prediction',name+'_absolute_error',name+'_squared_error']].to_numpy()).all()
  assert np.allclose(abs(wide[name+'_prediction']-wide.observed_Ea),wide[name+'_absolute_error'],rtol=1e-12,atol=1e-8)
 wide['prediction_disagreement']=wide[[name+'_prediction' for name in MODELS.values()]].std(axis=1,ddof=0)
 p3=wide[wide.protocol=='P3'].sort_values(['split_id','row_id']);p4=wide[wide.protocol=='P4'].sort_values(['split_id','row_id']);assert len(p3)==624 and len(p4)==183
 p3.to_csv(OUT/'p3_domain_prediction_instances.csv',index=False);p4.to_csv(OUT/'p4_domain_prediction_instances.csv',index=False)
 json_save(OUT/'domain_construction_status.json',{'plan_verified':True,'geometry_completed_before_loading_Ea':True,'predictive_models_fit':0,'P3_instances':len(p3),'P4_instances':len(p4)})
 print('Domain construction complete; no experimental model fit. PCA dimensions by fold:',[(r['split_id'],r['retained_PCs']) for r in fit_rows])
if __name__=='__main__':main()
