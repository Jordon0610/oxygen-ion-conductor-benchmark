"""Retrospective FE reevaluation. No fitting/tuning APIs. External target file parsed once."""
from pathlib import Path
import sys,json,hashlib,datetime,warnings
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,joblib
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from A2_applicability_domain.domain import chemistry
OUT=ROOT/'results/A3_2_external_validation';GROUP='combined_connected_group_id'
MODELS=['DummyRegressor','ExtraTreesRegressor','XGBoost','RBF-SVR','Ridge'];DUMMY='DummyRegressor';PRIMARY='ExtraTreesRegressor'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def jsave(n,o):
 with (OUT/n).open('x') as f:json.dump(o,f,ensure_ascii=False,indent=2);f.write('\n')
def csvsave(n,o):o.to_csv(OUT/n,index=False,float_format='%.17g')
def interval(a):return np.nanpercentile(a,[2.5,97.5]).tolist()
def draws(g,seed=320210):return np.random.default_rng(seed).integers(0,g,size=(10000,g))
def metric_arrays(frame,model,idx):
 gr=frame.groupby(GROUP,sort=True)[model+'_absolute_error'].agg(['mean','sum','count'])
 means=gr['mean'].to_numpy();sums=gr['sum'].to_numpy();sizes=gr['count'].to_numpy()
 return means[idx].mean(1),sums[idx].sum(1)/sizes[idx].sum(1)
def summarize_subset(frame,label,seed=320211,ci=True):
 n=frame[GROUP].nunique();idx=draws(n,seed) if ci else None
 result={'subset':label,'rows':len(frame),'groups':n,'group_ids':';'.join(sorted(frame[GROUP].unique()))}
 for model,short in [(PRIMARY,'ExtraTrees'),(DUMMY,'Dummy')]:
  a=frame[model+'_absolute_error'];result[short+'_macro_group_MAE_meV']=float(frame.groupby(GROUP)[model+'_absolute_error'].mean().mean());result[short+'_row_MAE_meV']=float(a.mean())
 result.update(ExtraTrees_RMSE_meV=float(np.sqrt(np.mean(frame[PRIMARY+'_squared_error']))),ExtraTrees_median_AE_meV=float(frame[PRIMARY+'_absolute_error'].median()),ExtraTrees_p90_AE_meV=float(frame[PRIMARY+'_absolute_error'].quantile(.9)))
 result['Delta_Dummy_macro_meV']=result['Dummy_macro_group_MAE_meV']-result['ExtraTrees_macro_group_MAE_meV'];result['relative_macro_improvement']=result['Delta_Dummy_macro_meV']/result['Dummy_macro_group_MAE_meV']
 result['Delta_Dummy_row_meV']=result['Dummy_row_MAE_meV']-result['ExtraTrees_row_MAE_meV']
 if ci:
  d,dr=metric_arrays(frame,DUMMY,idx);e,er=metric_arrays(frame,PRIMARY,idx)
  for name,v in [('ExtraTrees_macro',e),('ExtraTrees_row',er),('Delta_Dummy_macro',d-e),('Delta_Dummy_row',dr-er)]:result[name+'_CI_low'],result[name+'_CI_high']=interval(v)
  result['CI_caution']='fragile; few independent source groups'
 return result

def synthetic_tests():
 # Unequal group sizes must not silently change the primary estimand.
 f=pd.DataFrame({GROUP:['a','a','a','b'],PRIMARY+'_absolute_error':[1.,1.,1.,9.],DUMMY+'_absolute_error':[3.,3.,3.,10.],PRIMARY+'_squared_error':[1.,1.,1.,81.]})
 z=summarize_subset(f,'synthetic',ci=False);assert z['ExtraTrees_macro_group_MAE_meV']==5 and z['ExtraTrees_row_MAE_meV']==3
 idx=np.array([[0,0],[0,1],[1,1]]);a,b=metric_arrays(f,PRIMARY,idx);assert np.array_equal(a,[1,5,9]) and np.array_equal(b,[1,3,9])
 identical=f.copy();identical[DUMMY+'_absolute_error']=identical[PRIMARY+'_absolute_error'];a,_=metric_arrays(identical,DUMMY,idx);b,_=metric_arrays(identical,PRIMARY,idx);assert np.array_equal(a-b,[0,0,0])
 return {'unequal_group_macro_vs_row_weighting':True,'complete_group_multiplicity':True,'paired_zero_contrast':True}

def main():
 tests=synthetic_tests()
 assert not (OUT/'stage2_started.json').exists(), 'Current-FE evaluation already completed; rerun prohibited'
 assert sha(OUT/'analysis_plan_A3_2.json')==(OUT/'analysis_plan_A3_2.sha256').read_text().strip()
 assert sha(OUT/'final_model_manifest.json')==(OUT/'final_model_manifest.sha256').read_text().strip()
 plan=json.loads((OUT/'analysis_plan_A3_2.json').read_text());manifest=json.loads((OUT/'final_model_manifest.json').read_text())
 for p,h in manifest['frozen_artifact_hashes'].items():assert sha(OUT/p)==h,p
 for p,h in plan['source_hashes'].items():assert sha(ROOT/p)==h,p
 for m in manifest['models'].values():assert sha(ROOT/m['file'])==m['sha256']
 prior=json.loads((OUT/'prior_file_hashes.json').read_text())
 assert all((ROOT/p).exists() and sha(ROOT/p)==h for p,h in prior.items()),'Prior file changed before scoring'
 # Prediction and geometry transform only: every object is already frozen.
 X=pd.read_csv(OUT/'external_features_without_targets.csv',index_col=0,float_precision='round_trip')
 meta=pd.read_csv(OUT/'label_free_external_metadata.csv').set_index('external_record_id')
 preds={name:joblib.load(ROOT/manifest['models'][name]['file']).predict(X) for name in MODELS}
 assert all(len(p)==34 and np.isfinite(p).all() for p in preds.values())
 ref=json.loads((OUT/'domain_reference_manifest.json').read_text());art=np.load(OUT/'internal_domain_transform.npz');v=X[ref['composition_columns']].to_numpy();v[~np.isfinite(v)]=np.nan
 v=np.where(np.isnan(v),art['imputation_median'],v);keep=art['retained_feature_mask'];z=(v[:,keep]-art['training_mean'][keep])/art['training_std'][keep]
 pc=(z-art['PCA_mean'])@art['PCA_components'].T;dist=cdist(pc,art['train_PCs']);nearest=dist.argmin(1);d=dist.min(1);q=ref['q95_group']
 chemistry_ext=[chemistry(f) for f in meta.loc[X.index].normalized_formula];elements=set(ref['elements']);systems=set(ref['chemical_systems'])
 geometry=pd.DataFrame({'external_record_id':X.index,'nearest_training_row_id':[ref['training_ids'][i] for i in nearest],'nearest_training_composition_distance':d,'q95_group':q,'normalized_distance_group':d/q,'unseen_element':[bool(set(c[0])-elements) for c in chemistry_ext],'unseen_elements':[';'.join(sorted(set(c[0])-elements)) for c in chemistry_ext],'chemical_system':[c[1] for c in chemistry_ext],'unseen_chemical_system':[c[1] not in systems for c in chemistry_ext]})
 # Persist outcome-free predictions before the single label parse.
 unlabelled=pd.DataFrame({'external_record_id':X.index,**{m+'_prediction':p for m,p in preds.items()}});csvsave('frozen_predictions_before_label_access.csv',unlabelled)
 jsave('scoring_implementation_freeze.json',{'timestamp':now(),'script_sha256':sha(Path(__file__)),'synthetic_metric_tests':tests,'model_manifest_sha256':sha(OUT/'final_model_manifest.json'),'unlabelled_predictions_sha256':sha(OUT/'frozen_predictions_before_label_access.csv'),'stage1_complete':True,'external_target_access_count':0})
 jsave('stage2_started.json',{'timestamp':now(),'external_target_read_authorized_once':True,'all_model_and_preprocessing_fits_completed':True,'external_target_read_count_after_next_call':1})
 # THE ONLY FULL EXTERNAL LABEL READ IN THIS WORKFLOW.
 external=pd.read_csv(ROOT/plan['external_source'],float_precision='round_trip')
 assert len(external)==34 and external.external_record_id.tolist()==X.index.tolist()
 assert np.isfinite(external.Ea_meV.to_numpy(float)).all()
 table=external.merge(unlabelled,on='external_record_id',validate='one_to_one').merge(geometry,on='external_record_id',validate='one_to_one')
 table['observed_Ea_meV']=table.Ea_meV
 for m in MODELS:
  table[m+'_absolute_error']=abs(table[m+'_prediction']-table.observed_Ea_meV);table[m+'_squared_error']=(table[m+'_prediction']-table.observed_Ea_meV)**2
 table['source_evidence_limitation_A3_1d']=table.accounting_origin.eq('A3.1d_new')
 csvsave('external_prediction_table.csv',table)
 idx=draws(8);boot=pd.DataFrame({'bootstrap_draw':np.arange(10000),'sampled_group_indices':[','.join(map(str,r)) for r in idx]});results=[];bd,bdrow=metric_arrays(table,DUMMY,idx)
 for model in MODELS:
  bm,br=metric_arrays(table,model,idx);a=table[model+'_absolute_error'];macro=float(table.groupby(GROUP)[model+'_absolute_error'].mean().mean());base=float(table.groupby(GROUP)[DUMMY+'_absolute_error'].mean().mean())
  result={'model':model,'role':'primary' if model==PRIMARY else 'baseline' if model==DUMMY else 'secondary','N_rows':34,'N_groups':8,'macro_group_MAE_meV':macro,'row_weighted_MAE_meV':float(a.mean()),'RMSE_meV':float(np.sqrt(table[model+'_squared_error'].mean())),'median_absolute_error_meV':float(a.median()),'p90_absolute_error_meV':float(a.quantile(.9)),'Dummy_macro_group_MAE_meV':base,'Delta_Dummy_meV':base-macro,'relative_macro_improvement':(base-macro)/base,'Delta_Dummy_row_meV':float(table[DUMMY+'_absolute_error'].mean()-a.mean())}
  for n,values in [('macro_group_MAE',bm),('row_weighted_MAE',br),('Delta_Dummy',bd-bm),('Delta_Dummy_row',bdrow-br)]:result[n+'_CI_low'],result[n+'_CI_high']=interval(values)
  result['materiality_pass']=bool(result['Delta_Dummy_meV']>=25 and result['relative_macro_improvement']>=.1 and result['Delta_Dummy_CI_low']>0);result['CI_caution']='fragile, eight source-connected groups'
  results.append(result);boot[model+'_macro_MAE']=bm;boot[model+'_row_MAE']=br;boot[model+'_Delta_Dummy']=bd-bm
 csvsave('external_model_results.csv',pd.DataFrame(results));csvsave('group_bootstrap_results.csv',boot)
 groups=[]
 for group,f in table.groupby(GROUP,sort=True):
  for model in MODELS:
   mae=float(f[model+'_absolute_error'].mean());dm=float(f[DUMMY+'_absolute_error'].mean())
   groups.append({'group_id':group,'DOI':';'.join(sorted(f.DOI.unique())),'family':';'.join(sorted(f.structural_family.unique())),'N_rows':len(f),'bulk_rows':int(f.conductivity_component.eq('verified_bulk').sum()),'total_rows':int(f.conductivity_component.eq('verified_total').sum()),'observed_Ea_min_meV':float(f.observed_Ea_meV.min()),'observed_Ea_max_meV':float(f.observed_Ea_meV.max()),'model':model,'model_MAE_meV':mae,'Dummy_MAE_meV':dm,'model_minus_Dummy_meV':mae-dm,'source_evidence_limitation':bool(f.source_evidence_limitation_A3_1d.any())})
 csvsave('group_level_results.csv',pd.DataFrame(groups))
 dominance=[]
 for g in sorted(table[GROUP].unique()):
  s=summarize_subset(table[table[GROUP]!=g],'omit '+g,ci=False);s['omitted_group']=g;s['omitted_rows']=int(table[GROUP].eq(g).sum());dominance.append(s)
 csvsave('group_dominance_sensitivity.csv',pd.DataFrame(dominance))
 retained=table[~table.original_composition_overlap];s=summarize_subset(retained,'composition-novel source-external sensitivity');s['removed_rows']=len(table)-len(retained);s['removed_row_ids']=';'.join(table.loc[table.original_composition_overlap,'external_record_id']);s['remaining_formulas_independently_verified_absent_from_original483']=True;csvsave('composition_overlap_sensitivity.csv',pd.DataFrame([s]))
 csvsave('component_sensitivity.csv',pd.DataFrame([summarize_subset(f,c) for c,f in table.groupby('conductivity_component',sort=True)]))
 csvsave('family_descriptive_results.csv',pd.DataFrame([summarize_subset(f,c,ci=False) for c,f in table.groupby('structural_family',sort=True)]))
 excluded=plan['sensitivities']['source_evidence']['excluded_groups'];s=summarize_subset(table[~table[GROUP].isin(excluded)],'exclude two A3.1d SI-certified sources');s['excluded_groups']=';'.join(excluded);s['excluded_rows']=int(table[GROUP].isin(excluded).sum());csvsave('source_evidence_sensitivity.csv',pd.DataFrame([s]))
 domain_cols=['external_record_id',GROUP,'DOI','numeric_formula','normalized_formula','structural_family','conductivity_component','observed_Ea_meV',PRIMARY+'_prediction',PRIMARY+'_absolute_error']+list(geometry.columns[1:]);csvsave('external_domain_context.csv',table[domain_cols])
 # Frozen common cluster draws preserve all rows. Undefined correlations retained as NaN.
 units=[np.flatnonzero(table[GROUP].to_numpy()==g) for g in sorted(table[GROUP].unique())];rho=[]
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  for draw in idx:
   take=np.concatenate([units[i] for i in draw]);rho.append(float(spearmanr(table.normalized_distance_group.to_numpy()[take],table[PRIMARY+'_absolute_error'].to_numpy()[take]).statistic))
 estimate=float(spearmanr(table.normalized_distance_group,table[PRIMARY+'_absolute_error']).statistic);low,high=interval(rho)
 csvsave('domain_bootstrap_correlations.csv',pd.DataFrame({'bootstrap_draw':range(10000),'Spearman_rho':rho}))
 jsave('domain_context_summary.json',{'Spearman_rho':estimate,'CI_low':low,'CI_high':high,'valid_bootstrap_draws':int(np.isfinite(rho).sum()),'median_normalized_distance':float(table.normalized_distance_group.median()),'IQR_low':float(table.normalized_distance_group.quantile(.25)),'IQR_high':float(table.normalized_distance_group.quantile(.75)),'min_distance':float(table.normalized_distance_group.min()),'max_distance':float(table.normalized_distance_group.max()),'unseen_element_rows':int(table.unseen_element.sum()),'unseen_chemical_system_rows':int(table.unseen_chemical_system.sum()),'binary_external_eligibility_threshold_defined':False,'CI_caution':'fragile; only eight independent source groups'})
 jsave('stage2_completed.json',{'timestamp':now(),'external_raw_label_file_full_reads':1,'prediction_rows':34,'models_scored':MODELS,'groups_scored':8,'post_scoring_model_fits':0,'prediction_table_sha256':sha(OUT/'external_prediction_table.csv'),'classification':'EXTERNAL PREDICTIVE UTILITY SUPPORTED' if next(r for r in results if r['model']==PRIMARY)['materiality_pass'] else 'EXTERNAL PREDICTIVE UTILITY NOT ESTABLISHED'})
 print(pd.DataFrame(results)[['model','macro_group_MAE_meV','row_weighted_MAE_meV','Delta_Dummy_meV','Delta_Dummy_CI_low','Delta_Dummy_CI_high','materiality_pass']].to_string(index=False));print('Retrospective FE reevaluation complete. No fitting or tuning follows.')
if __name__=='__main__':
 if '--synthetic-test-only' in sys.argv:print(synthetic_tests())
 else:main()
