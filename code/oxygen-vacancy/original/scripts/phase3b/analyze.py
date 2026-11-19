"""Score completed outer predictions; paired inference from repeat-averaged losses."""
from common import *
MODELS=['ExtraTreesRegressor','XGBoost'];SARM=[f'shuffle_{i}' for i in range(1000,1020)];RARM=[f'random_{i}' for i in range(2000,2020)]
def save(name,df):df.to_csv(OUT/name,index=False,float_format='%.17g')
def collect():
 assert (OUT/'fit_completed.json').exists(),'No outer scoring until all prespecified fits finish'
 meta=pd.read_csv(ROOT/'results/benchmark/cohort_manifest.csv',float_precision='round_trip').set_index('row_id');rows=[];hp=[];auxrows=[]
 for c in read('experimental_configurations.json'):
  for arm in c['auxiliary_arms']:
   aux=json.loads((OUT/'auxiliary_cache'/(c['source_id']+'__'+arm+'.json')).read_text());z=dict(zip(aux['prediction_row_ids'],aux['predictions']))
   for role,ids in [('train',c['train_ids']),('test',c['test_ids'])]:
    for rid in ids:auxrows.append(dict(config_id=c['config_id'],split_id=c['split_id'],cleaning=c['cleaning'],source_id=c['source_id'],job_id=aux['job_id'],role=role,row_id=rid,arm=arm,auxiliary_model=aux['model'],auxiliary_target=aux['target'],control_seed=aux['seed'],auxiliary_prediction=z[rid],auxiliary_unit='synthetic standard-normal units' if arm.startswith('random_') else 'eV',source_formulas=len(aux['allowed_formula_ids'])))
  for model in c['downstream_models']:
   b=json.loads((OUT/'baseline_cache'/(c['config_id']+'__'+model+'.json')).read_text());a=json.loads((OUT/'downstream_cache'/(c['config_id']+'__'+model+'.json')).read_text())
   hp.append(dict(config_id=c['config_id'],split_id=c['split_id'],cleaning=c['cleaning'],downstream_model=model,selected_parameters=json.dumps(b['params'],sort_keys=True),candidate_MAE_meV=json.dumps(b['candidate_scores']),selection='baseline_only',N_train_rows=len(c['train_ids']),N_train_groups=meta.loc[c['train_ids'],'connected_group_id'].nunique(),N_test_rows=len(c['test_ids']),N_test_groups=meta.loc[c['test_ids'],'connected_group_id'].nunique()))
   predictions=[dict(arm='baseline',predictions=b['predictions'])]+a['arms']
   for item in predictions:
    for rid,pred in zip(c['test_ids'],item['predictions']):
     r=meta.loc[rid];obs=float(r.Ea)
     rows.append(dict(config_id=c['config_id'],split_id=c['split_id'],protocol=c['protocol'],cleaning=c['cleaning'],repeat=c['repeat'],fold=c['fold'],held_out_family=c['family'],downstream_model=model,arm=item['arm'],row_id=rid,connected_group_id=r.connected_group_id,family=r.family,conductivity_component=r.final_component,observed_Ea_meV=obs,predicted_Ea_meV=pred,absolute_error_meV=abs(obs-pred),squared_error_meV=(obs-pred)**2))
 ah=[]
 for job in read('auxiliary_jobs.json'):
  j=json.loads((OUT/'auxiliary_cache'/(job['job_id']+'.json')).read_text());ah.append(dict(job_id=j['job_id'],source_id=j['source_id'],arm=j['arm'],model=j['model'],target=j['target'],seed=j['seed'],N_allowed_formulas=len(j['allowed_formula_ids']),selected_parameters=json.dumps(j['params'],sort_keys=True),candidate_CV_MAE=json.dumps(j['candidate_scores']),target_hash=j['generated_target_hash']))
 save('auxiliary_hyperparameters.csv',pd.DataFrame(ah))
 p=pd.DataFrame(rows);save('outer_fold_predictions.csv',p);save('auxiliary_predictions.csv',pd.DataFrame(auxrows));save('downstream_hyperparameters.csv',pd.DataFrame(hp));return p

def interval(a):
 a=np.asarray(a);finite=a[np.isfinite(a)];return np.quantile(finite,[.025,.975]) if len(finite) else np.array([np.nan,np.nan])

def context_analysis(q,name,universe=None):
 # One independent outcome per original row-arm: loss averaged over repeats, never averaged prediction loss.
 reduced=q.groupby(['row_id','connected_group_id','conductivity_component','arm'],sort=True).agg(loss=('absolute_error_meV','mean'),square=('squared_error_meV','mean'),observed=('observed_Ea_meV','first'),N_predictions=('repeat','size')).reset_index()
 loss=reduced.pivot(index='row_id',columns='arm',values='loss').sort_index();square=reduced.pivot(index='row_id',columns='arm',values='square').loc[loss.index];info=reduced.drop_duplicates('row_id').set_index('row_id').loc[loss.index]
 assert not loss.isna().any().any();assert set(SARM+RARM+['baseline','real_Efmin'])<=set(loss.columns)
 loss['mean_shuffled_loss']=loss[SARM].mean(axis=1);loss['mean_random_loss']=loss[RARM].mean(axis=1)
 groups=sorted(info.connected_group_id.unique()) if universe is None else sorted(universe);g=len(groups);actual=info.connected_group_id.nunique();bootok=actual>=5
 ns=info.groupby('connected_group_id').size().reindex(groups,fill_value=0).to_numpy();sums=loss.groupby(info.connected_group_id).sum().reindex(groups,fill_value=0)
 weights=np.random.default_rng(8675309).multinomial(g,np.ones(g)/g,size=5000);den=weights@ns;boot=np.divide(weights@sums.to_numpy(),den[:,None],out=np.full((5000,len(loss.columns)),np.nan),where=den[:,None]>0);boot=pd.DataFrame(boot,columns=loss.columns)
 armrows=[];basemean=loss.baseline.mean();observed=info.observed.to_numpy()
 for arm in loss:
  mae=loss[arm].mean();lo,hi=interval(boot[arm]) if bootok else [np.nan,np.nan];delta=basemean-mae;di=interval(boot.baseline-boot[arm]) if bootok else [np.nan,np.nan]
  rmse=float(np.sqrt(square[arm].mean())) if arm in square else np.nan
  r2=float(1-square[arm].sum()/np.sum((observed-observed.mean())**2)) if arm in square and len(observed)>=20 and np.std(observed)>=100 else np.nan
  armrows.append(dict(context=name,arm=arm,N_rows=len(loss),N_groups=actual,N_bootstrap_source_groups=g,MAE_meV=mae,MAE_CI_low=lo,MAE_CI_high=hi,RMSE_meV=rmse,R2=r2,baseline_minus_arm_meV=delta,delta_CI_low=di[0],delta_CI_high=di[1],relative_improvement=delta/basemean,valid_bootstrap_draws=int(np.isfinite(boot[arm]).sum()) if bootok else 0))
 contrasts=[]
 for label,left,right in [('baseline_minus_real','baseline','real_Efmin'),('mean_shuffle_minus_real','mean_shuffled_loss','real_Efmin'),('mean_random_minus_real','mean_random_loss','real_Efmin'),('baseline_minus_Efmean','baseline','real_Efmean'),('baseline_minus_XGBaux','baseline','real_Efmin_XGBaux')]:
  if right not in loss:continue
  delta=float((loss[left]-loss[right]).mean());lo,hi=interval(boot[left]-boot[right]) if bootok else [np.nan,np.nan];rel=delta/float(loss[left].mean());ri=interval((boot[left]-boot[right])/boot[left]) if bootok else [np.nan,np.nan]
  contrasts.append(dict(context=name,contrast=label,left_arm=left,right_arm=right,N_rows=len(loss),N_groups=actual,N_bootstrap_source_groups=g,left_MAE_meV=float(loss[left].mean()),right_MAE_meV=float(loss[right].mean()),delta_meV=delta,delta_CI_low=lo,delta_CI_high=hi,relative_improvement=rel,relative_CI_low=ri[0],relative_CI_high=ri[1],material_positive=bool(bootok and delta>=25 and rel>=.1 and lo>0),material_negative=bool(bootok and delta<=-25 and rel<=-.1 and hi<0),valid_bootstrap_draws=int(np.isfinite(boot[left]-boot[right]).sum()) if bootok else 0))
 result=pd.DataFrame(contrasts);primary=result[result.contrast.eq('baseline_minus_real')].iloc[0];sh=result[result.contrast.eq('mean_shuffle_minus_real')].iloc[0];ra=result[result.contrast.eq('mean_random_minus_real')].iloc[0]
 summary=dict(context=name,N_rows=len(loss),N_groups=actual,baseline_MAE_meV=basemean,real_MAE_meV=float(loss.real_Efmin.mean()),delta_real_meV=primary.delta_meV,relative_improvement=primary.relative_improvement,delta_CI_low=primary.delta_CI_low,delta_CI_high=primary.delta_CI_high,material_positive=bool(primary.material_positive),material_negative=bool(primary.material_negative),mean_shuffled_MAE_meV=float(loss.mean_shuffled_loss.mean()),mean_random_MAE_meV=float(loss.mean_random_loss.mean()),shuffle_specific_delta_meV=sh.delta_meV,shuffle_specific_CI_low=sh.delta_CI_low,shuffle_specific_CI_high=sh.delta_CI_high,random_specific_delta_meV=ra.delta_meV,random_specific_CI_low=ra.delta_CI_low,random_specific_CI_high=ra.delta_CI_high,shuffled_seeds_worse_than_real=int((loss[SARM].mean()>loss.real_Efmin.mean()).sum()),random_seeds_worse_than_real=int((loss[RARM].mean()>loss.real_Efmin.mean()).sum()),primary_and_specific=bool(primary.material_positive and sh.delta_CI_low>0 and ra.delta_CI_low>0),interval_status='conditional cluster interval; fragile with few groups' if bootok else 'exploratory point estimate only; G<5')
 seeds=[]
 for arm in SARM+RARM:
  ar=next(r for r in armrows if r['arm']==arm);seeds.append(dict(**ar,control_type='shuffle' if arm in SARM else 'random',seed=int(arm.split('_')[1]),MAE_worse_than_real=bool(loss[arm].mean()>loss.real_Efmin.mean()),control_minus_real_meV=float((loss[arm]-loss.real_Efmin).mean())))
 groupdiff=pd.DataFrame({'row_id':loss.index,'group':info.connected_group_id,'baseline_loss':loss.baseline,'real_loss':loss.real_Efmin,'improvement':loss.baseline-loss.real_Efmin}).groupby('group').agg(N_rows=('row_id','size'),baseline_MAE=('baseline_loss','mean'),real_MAE=('real_loss','mean'),mean_improvement_meV=('improvement','mean')).reset_index();groupdiff['context']=name
 return pd.DataFrame(armrows),result,summary,pd.DataFrame(seeds),groupdiff,reduced.assign(context=name)

def analyze(p):
 reportids=read('aggregate_reporting_row_ids.json');contexts=[]
 for cleaning,model,label in [('exact_clean','ExtraTreesRegressor','primary_exact_ET'),('system_clean','ExtraTreesRegressor','system_clean_ET'),('exact_clean','XGBoost','secondary_downstream_XGB')]:
  q=p[p.protocol.eq('P3')&p.cleaning.eq(cleaning)&p.downstream_model.eq(model)&p.row_id.isin(reportids[cleaning])];assert q.groupby(['row_id','arm'])['repeat'].nunique().eq(2).all();contexts.append((q,label,None))
 for family in ['Perovskite-related','Apatite','Scheelite','δ-Bi2O3']:
  q=p[p.protocol.eq('P4')&p.held_out_family.eq(family)&p.downstream_model.eq('ExtraTreesRegressor')];contexts.append((q,'P4_'+family,None))
  if family=='Perovskite-related':
   for component in ['bulk','total']:contexts.append((q[q.conductivity_component.eq(component)],'Perovskite_'+component,set(q.connected_group_id)))
 arms=[];contrasts=[];summaries=[];seeds=[];groupdiff=[];losses=[]
 for q,name,universe in contexts:
  a,c,s,n,g,r=context_analysis(q,name,universe);arms.append(a);contrasts.append(c);summaries.append(s);seeds.append(n);groupdiff.append(g);losses.append(r)
 arms=pd.concat(arms);contrasts=pd.concat(contrasts);summaries=pd.DataFrame(summaries);seeds=pd.concat(seeds)
 save('all_context_arm_metrics.csv',arms);save('paired_transfer_contrasts.csv',contrasts);save('all_context_transfer_summary.csv',summaries);save('null_control_seed_results.csv',seeds);save('per_group_transfer_effects.csv',pd.concat(groupdiff));save('repeat_averaged_row_losses.csv',pd.concat(losses))
 save('aggregate_transfer_results.csv',arms[arms.context.eq('primary_exact_ET')]);save('system_clean_sensitivity.csv',summaries[summaries.context.eq('system_clean_ET')]);save('downstream_model_sensitivity.csv',summaries[summaries.context.isin(['primary_exact_ET','secondary_downstream_XGB'])]);save('auxiliary_model_sensitivity.csv',contrasts[contrasts.context.eq('primary_exact_ET')&contrasts.contrast.isin(['baseline_minus_real','baseline_minus_XGBaux','baseline_minus_Efmean'])]);save('perovskite_related_transfer.csv',summaries[summaries.context.eq('P4_Perovskite-related')]);save('perovskite_component_sensitivity.csv',summaries[summaries.context.isin(['Perovskite_bulk','Perovskite_total'])]);save('exploratory_family_results.csv',summaries[summaries.context.isin(['P4_Apatite','P4_Scheelite','P4_δ-Bi2O3'])])
 by=summaries.set_index('context');primary=by.loc['primary_exact_ET'];robust_negative=bool(by.loc[['system_clean_ET','secondary_downstream_XGB'],'material_negative'].any())
 if primary.material_negative:category='NEGATIVE TRANSFER'
 elif not primary.material_positive:category='NO MATERIAL TRANSFER'
 elif not primary.primary_and_specific:category='NON-SPECIFIC AUXILIARY BENEFIT'
 elif not robust_negative:category='DEFECT-SPECIFIC POSITIVE TRANSFER'
 else:category='NO MATERIAL TRANSFER'
 js(OUT/'decision_gate.json',dict(category=category,primary_material_positive=bool(primary.material_positive),primary_material_negative=bool(primary.material_negative),beats_shuffle_CI=bool(primary.shuffle_specific_CI_low>0),beats_random_CI=bool(primary.random_specific_CI_low>0),system_direction_positive=bool(by.loc['system_clean_ET','delta_real_meV']>0),secondary_downstream_direction_positive=bool(by.loc['secondary_downstream_XGB','delta_real_meV']>0),major_robustness_reversal=robust_negative,Perovskite_material_positive=bool(by.loc['P4_Perovskite-related','material_positive']),broad_cross_family_claim_authorized=False))
 print(summaries[['context','N_rows','N_groups','baseline_MAE_meV','real_MAE_meV','delta_real_meV','delta_CI_low','delta_CI_high','material_positive','shuffle_specific_delta_meV','random_specific_delta_meV']].to_string(index=False));print(category)
if __name__=='__main__':plan();analyze(collect())
