"""Current FE aggregation using unchanged Phase2 estimands and bootstrap functions."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_benchmark_r9 import verify_lock,save
from benchmark.summarize import load_predictions,per_row,aggregate,paired,intervals,cluster_draws,bounds
from benchmark.design import MODELS
OUT=ROOT/'results/benchmark'
def main():
 verify_lock();assert (ROOT/'jobs/benchmark_fitting_completed.json').exists()
 m=pd.read_csv(OUT/'cohort_manifest.csv').set_index('row_id');hmeta=pd.read_csv(ROOT/'support/historical_benchmark/cohort_manifest.csv').set_index('row_id')
 fields=['family','final_component','normalized_doi','connected_group_id','quality_flag','Ea']
 assert all(m[k].fillna('').astype(str).tolist()==hmeta.loc[m.index,k].fillna('').astype(str).tolist() for k in fields),'Frozen source/family/component/target changed beyond accepted revision'
 save(ROOT/'support/frozen_metadata_identity.json',{'all483_families_components_sources_groups_quality_targets_identical':True,'Perovskite_and_Perovskite_related_are_separate_frozen_labels':True,'R0053_family':m.loc['R0053','family'],'R0023_family':m.loc['R0023','family']})
 p=load_predictions();p['result_version']='GLOBAL_FE_R9';p.to_csv(OUT/'outer_predictions.csv',index=False,float_format='%.17g')
 d=per_row(p);d['result_version']='GLOBAL_FE_R9';d.to_csv(OUT/'per_record_losses.csv',index=False,float_format='%.17g')
 expected={c:{'P1':n*2*5,'P2':n*2*5,'P3':n*2*5} for c,n in [('K_FE',312),('B_FE',155),('T',157)]};expected['KQ_FE']={'P3':308*2*5}
 actual=p.groupby(['cohort','protocol']).size().to_dict()
 for c,pp in expected.items():
  for protocol,n in pp.items():assert actual[(c,protocol)]==n,(c,protocol,actual[(c,protocol)],n)
 assert len(p[(p.cohort=='K_FE')&p.protocol.isin(['P1','P2','P3'])])==9360
 assert not p.duplicated(['split_id','model','row_id']).any()
 coverage=pd.read_csv(OUT/'confirmatory_family_manifest.csv');result=[];families=[]
 (OUT/'bootstrap_weights').mkdir(exist_ok=True)
 for (cohort,protocol,model),x in d.groupby(['cohort','protocol','model'],sort=False):
  allowed=set(coverage.loc[(coverage.cohort==cohort)&coverage.P4_eligible,'family'])
  for scope,z in [('all_tested',x),('matched_confirmatory',x[x.family.isin(allowed)])]:
   if z.empty:continue
   for macro in ([False,True] if protocol=='P4' else [False]):
    result.append({'cohort':cohort,'protocol':protocol,'model':model,'scope':scope,**aggregate(z,macro,protocol=='P4'),'mean_training_rows':z.training_rows.mean(),'mean_training_groups':z.training_groups.mean(),'result_version':'GLOBAL_FE_R9','output_status':'REUSE_IDENTICAL_DEPENDENCIES' if cohort=='T' else 'REFRESHED_CURRENT_DEPENDENCIES'})
  if protocol=='P4':
   for fam,z in x.groupby('family'):
    families.append({'cohort':cohort,'model':model,'family':fam,**aggregate(z,False,True),'training_rows':int(z.training_rows.iloc[0]),'training_groups':int(z.training_groups.iloc[0]),'other_family_rows_purged':int(z.other_family_rows_purged.iloc[0]),'status':'success','result_version':'GLOBAL_FE_R9'})
  if model=='ExtraTreesRegressor':
   w=cluster_draws(x,protocol=='P4')
   if w is not None:np.savez_compressed(OUT/'bootstrap_weights'/(cohort+'_'+protocol+'.npz'),multiplicity=w,row_ids=x.row_id.to_numpy(),groups=x.connected_group_id.to_numpy(),families=x.family.to_numpy(),seed=np.array(8675309))
 pd.DataFrame(result).to_csv(OUT/'model_results.csv',index=False,float_format='%.17g');pd.DataFrame(families).to_csv(OUT/'family_ood_results.csv',index=False,float_format='%.17g')
 contrasts=[];sensitivity=[];quality=[];global_dependency=[]
 for model in MODELS:
  for cohort in ['K_FE','B_FE','T']:
   c=d[(d.model==model)&(d.cohort==cohort)]
   for new,ref in [('P2','P1'),('P3','P2'),('P3','P1')]:
    row={'cohort':cohort,'model':model,**paired(c[c.protocol==new],c[c.protocol==ref],new+' minus '+ref),'result_version':'GLOBAL_FE_R9'};contrasts.append(row)
    if (new,ref)==('P3','P1'):sensitivity.append({'analysis':'within-component protocol shift',**row})
   a=c[c.protocol=='P4'];b=c[c.protocol=='P3']
   if not a.empty:
    row={'cohort':cohort,'model':model,**paired(a,b,'P4 minus P3, matched family rows',True),'result_version':'GLOBAL_FE_R9'};contrasts.append(row);sensitivity.append({'analysis':'within-component family shift',**row})
    global_dependency.append({'cohort':cohort,'model':model,**paired(a,b,'P4 minus P3, matched family rows',False),'bootstrap_unit':'unique global connected group without family strata','result_version':'GLOBAL_FE_R9'})
  for cohort,comp in [('B_FE','bulk'),('T','total')]:
   for protocol in ['P3','P4']:
    a=d[(d.model==model)&(d.cohort==cohort)&(d.protocol==protocol)];b=d[(d.model==model)&(d.cohort=='K_FE')&(d.protocol==protocol)&(d.component==comp)]
    sensitivity.append({'analysis':'component-only training vs current K on shared records','cohort':cohort,'model':model,**paired(a,b,protocol+': '+cohort+' minus K_FE on shared '+comp+' rows',protocol=='P4'),'result_version':'GLOBAL_FE_R9'})
  for protocol in ['P3','P4']:
   a=d[(d.model==model)&(d.cohort=='KQ_FE')&(d.protocol==protocol)];b=d[(d.model==model)&(d.cohort=='K_FE')&(d.protocol==protocol)]
   quality.append({'model':model,'protocol':protocol,**paired(a,b,'KQ_FE minus K_FE on shared quality-eligible rows',protocol=='P4'),'result_version':'GLOBAL_FE_R9'})
 for n,rr in [('protocol_comparisons.csv',contrasts),('component_sensitivity.csv',sensitivity),('quality_sensitivity.csv',quality),('global_group_dependency_sensitivity.csv',global_dependency)]:pd.DataFrame(rr).to_csv(OUT/n,index=False,float_format='%.17g')
 # H on identical FE test records is a distinct comparator, not headline subtraction or local R5.
 hp=pd.read_csv(ROOT/'support/historical_benchmark/outer_predictions.csv',float_precision='round_trip');hd=per_row(hp);hisrows=pd.read_csv(ROOT/'support/historical_benchmark/model_results.csv');vers=[];hc=[]
 mapping={'K_FE':'K','B_FE':'B','T':'T','KQ_FE':'KQ'}
 for (c,protocol,model),cur in d.groupby(['cohort','protocol','model'],sort=False):
  hh=hd[(hd.cohort==mapping[c])&(hd.protocol==protocol)&(hd.model==model)];same=hh[hh.row_id.isin(cur.row_id)]
  assert set(same.row_id)==set(cur.row_id)
  for label,dat in [('H_ON_ALL_HISTORICAL_TEST_ROWS',hh),('H_ON_COMMON_FE_TEST_ROWS',same),('CURRENT_FE_TEST_ROWS',cur)]:
   vers.append({'cohort':c,'protocol':protocol,'model':model,'version_scope':label,**aggregate(dat,False,protocol=='P4'),'interpretation':'Historical/current paired comparison includes changed training/selection; not a causal cleaning effect or matched local R5 estimand'})
  hc.append({'cohort':c,'protocol':protocol,'model':model,**paired(cur,same,'FE minus H on identical current evaluation rows',protocol=='P4'),'result_version':'GLOBAL_FE_VS_H_COMMON_ROWS','not_pure_cleaning_effect':True})
 pd.DataFrame(vers).to_csv(ROOT/'04_HISTORICAL_VS_CURRENT_SUMMARY.csv',index=False,float_format='%.17g');pd.DataFrame(hc).to_csv(OUT/'FE_vs_H_common_test_paired_contrasts.csv',index=False,float_format='%.17g')
 # R5 binding: exact source-excluded FE selection/training memberships must match before comparing predictions.
 local=pd.read_csv(ROOT/'support/accepted_R5/03_MATCHED_PARENT_PREDICTIONS.csv',float_precision='round_trip');parity=[]
 for rep in [0,1]:
  q=p[(p.cohort=='K_FE')&(p.protocol=='P3')&(p.repeat==rep)&p.row_id.isin([f'R{i:04d}' for i in range(1,39)])]
  for name in ['DummyRegressor','ExtraTreesRegressor','XGBoost']:
   cur=q[q.model==name];old=local[(local.arm=='FE_MATCHED')&(local.model==name)&(local.repeat==rep)]
   a=cur.set_index('row_id').y_pred;b=old.set_index('row_id').prediction_meV
   assert set(a.index)==set(b.index);maxdiff=float(np.max(np.abs(a.loc[b.index].to_numpy()-b.to_numpy())))
   sid=cur.split_id.unique().tolist();assert len(sid)==1
   cp=json.loads((OUT/'checkpoints'/(sid[0]+'__'+name+'.json')).read_text())
   parity.append({'repeat':rep,'model':name,'current_split_id':sid[0],'historical_split_id':cp['historical_split_id'],'parents':len(a),'max_abs_prediction_difference_meV':maxdiff,'locked_tolerance_meV':1e-6,'parity':maxdiff<=1e-6,'R9_selected_parameters':json.dumps(cp['selected_parameters'],sort_keys=True),'meaning':'Only identical local FE recipes can be parity-bound; not global H_MATCHED relabelling'})
 assert all(x['parity'] for x in parity),parity
 pd.DataFrame(parity).to_csv(OUT/'R5_FE_global_P3_binding.csv',index=False,float_format='%.17g')
 # Current decisions from current outputs under inherited thresholds, never forced to historical labels.
 c=pd.DataFrame(contrasts);k=c[(c.cohort=='K_FE')&c.model.isin(MODELS[1:])]
 decisions={label:{'predictive_models_supported':int(k[k.comparison==label].supported_material_increase.sum()),'criterion':'>=25meV,>=10% reference MAE,positive group-bootstrap lower bound;>=3/4 predictive models'} for label in ['P3 minus P1','P4 minus P3, matched family rows']}
 save(OUT/'decision_gate.json',decisions)
 save(ROOT/'jobs/benchmark_summary_completed.json',{'OOF_counts':{c+'/'+pr:int(n) for (c,pr),n in actual.items()},'primary_K_FE_instances':9360,'finite_predictions_retained':True,'R5_parity_checks':parity,'T_only_complete_dependency_reuse':True,'P4_current_rows':int(d[(d.cohort=='K_FE')&(d.protocol=='P4')&(d.model=='ExtraTreesRegressor')].row_id.nunique()),'P4_current_unique_global_groups':int(d[(d.cohort=='K_FE')&(d.protocol=='P4')&(d.model=='ExtraTreesRegressor')].connected_group_id.nunique())})
 verify_lock();print('Current benchmark summaries, bootstrap weights, common-row H comparisons and R5 bindings saved.',flush=True)
if __name__=='__main__':main()
