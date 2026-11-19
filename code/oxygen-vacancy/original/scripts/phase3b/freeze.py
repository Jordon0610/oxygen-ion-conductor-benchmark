from pathlib import Path
import hashlib,json,datetime
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/phase3b'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(name,d):(OUT/name).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
assert not (OUT/'analysis_plan_phase3b.json').exists()
for f in ['results/phase3a_2/prior_file_hashes.json','results/phase3a_2/deliverable_hashes.json']:
 for p,h in json.loads((ROOT/f).read_text()).items():assert sha(ROOT/p)==h,p
paths=set()
for folder in ['data','results','scripts','sources','tests']:
 for p in (ROOT/folder).rglob('*'):
  if p.is_file() and 'phase3b' not in p.parts and '__pycache__' not in p.parts and p.name!='.DS_Store':paths.add(p)
for p in ROOT.iterdir():
 if p.is_file() and p.suffix in ['.md','.txt','.json','.toml','.lock']:paths.add(p)
snapshot={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)};js('prior_file_hashes.json',snapshot)
p2=json.loads((ROOT/'results/benchmark/analysis_plan_v1.json').read_text());p31=json.loads((ROOT/'results/phase3a_1/analysis_plan_phase3a_1.json').read_text())
el=pd.read_csv(ROOT/'results/phase3a_2/experimental_row_fold_eligibility.csv');meta=pd.read_csv(ROOT/'results/phase3a_1/experimental_metadata_only.csv').set_index('row_id')
splits={s['split_id']:s for s in json.loads((ROOT/'results/benchmark/splits.json').read_text()) if s['cohort']=='K' and s['protocol'] in ['P3','P4']}
rem=pd.read_csv(ROOT/'results/phase3a_2/dft_pretraining_exclusion_manifest.csv');configs=[];manifest=[]
for (sid,cleaning),q in el.groupby(['split_id','cleaning'],sort=True):
 s=splits[sid]
 if s['protocol']=='P4' and cleaning!='exact_clean':continue
 tr=sorted(q[q.role.eq('train')&q.in_domain].row_id);te=sorted(q[q.role.eq('test')&q.in_domain].row_id)
 inner=[]
 for i,(a,b) in enumerate(GroupKFold(3,shuffle=True,random_state=s['seed']+100+s['fold']).split(tr,groups=meta.loc[tr,'connected_group_id'])):
  inner.append(dict(inner_fold=i,train_ids=[tr[j] for j in a],validation_ids=[tr[j] for j in b]))
 rr=rem[rem.split_id.eq(sid)&rem.cleaning.eq(cleaning)];allowed=sorted(rr[~rr.removed].formula_id);popid=hashlib.sha256('\n'.join(allowed).encode()).hexdigest()[:16]
 cfg=dict(config_id=sid+'__'+cleaning,split_id=sid,cleaning=cleaning,protocol=s['protocol'],repeat=s['repeat'],fold=s['fold'],family=s['held_out_family'],train_ids=tr,test_ids=te,original_purged_ids=s['purged_ids'],inner_splits=inner,allowed_DFT_formula_ids=allowed,source_id=popid,
 downstream_models=['ExtraTreesRegressor','XGBoost'] if cleaning=='exact_clean' and s['protocol']=='P3' else ['ExtraTreesRegressor'])
 configs.append(cfg)
 for role,ids in [('train',tr),('test',te)]:
  for rid in ids:manifest.append(dict(config_id=cfg['config_id'],split_id=sid,cleaning=cleaning,protocol=s['protocol'],repeat=s['repeat'],fold=s['fold'],role=role,row_id=rid,connected_group_id=meta.loc[rid,'connected_group_id'],family=meta.loc[rid,'family'],component=meta.loc[rid,'final_component']))
manifest=pd.DataFrame(manifest);manifest.to_csv(OUT/'frozen_eligible_membership.csv',index=False)
# Exact pairing and system-clean always-eligible reporting population are frozen without Ea.
intersections={}
for cleaning in ['exact_clean','system_clean']:
 q=manifest[manifest.protocol.eq('P3')&manifest.cleaning.eq(cleaning)&manifest.role.eq('test')];count=q.groupby('row_id')['repeat'].nunique();ids=sorted(count[count==2].index);intersections[cleaning]=ids
 assert len(ids)==(192 if cleaning=='exact_clean' else 184)
 assert meta.loc[ids,'connected_group_id'].nunique()==(32 if cleaning=='exact_clean' else 30)
js('aggregate_reporting_row_ids.json',intersections);js('experimental_configurations.json',configs)
jobs={}
for c in configs:
 arms=[('real_Efmin','ExtraTreesRegressor','Ef_min_sourcefiltered_q0',None)]+[(f'shuffle_{i}','ExtraTreesRegressor','Ef_min_sourcefiltered_q0',i) for i in range(1000,1020)]+[(f'random_{i}','ExtraTreesRegressor','Ef_min_sourcefiltered_q0',i) for i in range(2000,2020)]
 if c['cleaning']=='exact_clean':arms.append(('real_Efmean','ExtraTreesRegressor','Ef_mean_sourcefiltered_q0',None))
 if c['cleaning']=='exact_clean' and c['protocol']=='P3':arms.append(('real_Efmin_XGBaux','XGBoost','Ef_min_sourcefiltered_q0',None))
 c['auxiliary_arms']=[a[0] for a in arms]
 for arm,model,target,seed in arms:
  jid=c['source_id']+'__'+arm
  if jid not in jobs:jobs[jid]=dict(job_id=jid,source_id=c['source_id'],arm=arm,model=model,target=target,seed=seed,allowed_formula_ids=c['allowed_DFT_formula_ids'],prediction_row_ids=[],configuration_ids=[])
  jobs[jid]['prediction_row_ids']=sorted(set(jobs[jid]['prediction_row_ids'])|set(c['train_ids']+c['test_ids']));jobs[jid]['configuration_ids'].append(c['config_id'])
js('experimental_configurations.json',configs);js('auxiliary_jobs.json',list(jobs.values()))
plan={
'version':1,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'frozen_before_any_Phase3B_fit':True,
'source_hashes':snapshot,'target':{'experimental':'primary Ea, meV; literature-reported apparent conductivity activation energy in oxygen-ion-conducting oxides','source':'unchanged Cohort K primary Ea in Phase2 cohort_manifest; no HT regimes','DFT_primary':'Ef_min_sourcefiltered_q0','DFT_secondary':'Ef_mean_sourcefiltered_q0','DFT_population':'published charge0.csv only, 1745 labels / 824 formulas; pinned hash 12f2f270127f98be9f1dc5e73619f1c995ab21b71f3a94493081d5890f37e596'},
'frozen_membership':{'file':'frozen_eligible_membership.csv','sha256':sha(OUT/'frozen_eligible_membership.csv'),'configurations_sha256':sha(OUT/'experimental_configurations.json'),'reporting_row_ids_sha256':sha(OUT/'aggregate_reporting_row_ids.json'),'exact':'192 unique rows / 32 original groups, both repeats, ten original P3 folds','system':'fold-specific train/test eligibility unchanged; aggregate inference restricted to always-eligible 184 rows / 30 original groups; predictions for other eligible fold rows retained'},
'features':{'downstream':'exact frozen Phase2 composition_features.csv, all 144 columns including fixed component indicators; baseline unmodified; each auxiliary adds one scalar','DFT':'exact frozen Phase3A.1 139 composition-only features','prohibited':['family','DOI','distance','group ID','residuals','HT metadata'],'preprocessing':'median imputation (empty ->0) and training-only exact-constant filter for both tree estimators, no scaling; no global fitted preprocessing'},
'cleaning':'reuse Phase3A.2 full-483 test-group closure exclusions and allowed source lists; no recomputed outer eligibility; exact or nonO chemical system as frozen',
'auxiliary_policy':{'primary':'ExtraTreesRegressor','secondary':'XGBoost real Ef_min exact P3 only','candidates':{k:p31['models'][k] for k in ['ExtraTreesRegressor','XGBoost']},'fixed':p31['fixed_model_parameters'],'seed':161803,'selection':'five chemical-system GroupKFold folds, shuffle=True, seed314159; sorted formula IDs, sorted factorized chemical systems; pooled formula MAE; first candidate wins ties; refit all allowed formulas','null_generation':'per allowed population sorted formula IDs; numpy default_rng(seed).permutation(formula Ef_min) for shuffled, standard_normal(N) for random; generate once before DFT CV; same fixed vector across candidates; no experimental Ea available','shuffle_seeds':list(range(1000,1020)),'random_seeds':list(range(2000,2020)),'cache':'identical allowed IDs, target construction, seed, estimator and grid share deterministic fit; every configuration links to its job; never cache across different allowed sets','jobs_sha256':sha(OUT/'auxiliary_jobs.json'),'number_unique_jobs':len(jobs)},
'downstream_policy':{'primary':'ExtraTreesRegressor','secondary':'XGBoost exact aggregate P3','candidates':{k:p2['hyperparameter_search_spaces'][k] for k in ['ExtraTreesRegressor','XGBoost']},'fixed':{k:p2['fixed_hyperparameters'][k] for k in ['ExtraTreesRegressor','XGBoost']},'inner':'three shuffled GroupKFold connected-group splits on eligible outer-training rows; seed original outer seed +100+fold; frozen before outcomes; pooled row MAE, first candidate wins ties','selection':'composition baseline only; finish every baseline selection before auxiliary fitting/evaluation; reuse selected hyperparameters and model seed across all matched arms','inner_auxiliary_fits':'none: latest user mandates baseline-only tuning, so no DFT-supervised feature enters inner validation or model selection; outer eligibility remains frozen; this differs from an augmented-arm inner-tuning design that would require inner DFT exclusions','outer_fit':'all arms identical eligible train/test IDs; test Ea unavailable to fitting/tuning functions; load outcomes for scoring only after predictions saved'},
'analyses':{'primary':'exact P3, DFT ExtraTrees -> experimental ExtraTrees: baseline, real Efmin,20shuffled,20random','system_sensitivity':'system P3 same A/B/C/D, report both-repeat intersection','secondary_downstream':'exact P3 downstream XGBoost same controls and Efmean','secondary_auxiliary':'exact P3 real XGBoost auxiliary into downstream ExtraTrees, compare same baseline','Efmean':'secondary exact P3 and exact P4, never promotes to primary','P4':'Perovskite-related only confirmatory 40 rows/11groups; Apatite37original/28covered/3groups, Scheelite16covered/4groups,delta12covered/4groups exploratory; all four frozen P4 train/test/purges reused; no macro-family confirmation','Perovskite_component':'same family-trained predictions stratified by bulk16rows/6groups and total24rows/7groups, not separate component model refits; two groups overlap; sensitivity only'},
'uncertainty':{'bootstrap_seed':8675309,'draws':5000,'unit':'original global connected_group_id, common multiplicity across all arms','repeat_handling':'average each row-arm absolute loss across its two OOF predictions before bootstrap; not loss of averaged prediction; P4 one prediction','estimand':'row-weighted MAE; paired baseline-real, null ensemble-real, auxiliary-secondary gains','null_ensemble':'mean absolute error across all20 seeds per row, after repeat averaging; never average predictions to obtain ensemble error','CI':'percentile95%, conditional on fitted models/frozen source and splits; resample G groups with replacement retaining all rows; P4G<5 report point estimates without inferential CI; Perovskite componentG6/7 fragile intervals','additional_metrics':'RMSE computed from repeat-averaged squared losses; R2 descriptive only N>=20 and target SD>=100meV; no target changes'},
'criteria':{'material_positive':'baseline MAE-real MAE >=25meV AND relative gain>=0.10 AND paired delta CI lower>0','material_negative':'real MAE-baseline MAE>=25meV AND relative worsening>=0.10 AND paired baseline-real delta CI upper<0','defect_specific':'material positive AND shuffled-mean-loss minus real CI lower>0 AND random-mean-loss minus real CI lower>0','robustness':'system-clean supportive iff delta>0; secondary model direction reported; major contradictory reversal = symmetric material_negative criterion in system or downstream sensitivity, not merely negative tiny point estimate','categories_priority':['NEGATIVE TRANSFER if primary material_negative','NO MATERIAL TRANSFER if primary material_positive fails','NON-SPECIFIC AUXILIARY BENEFIT if primary material_positive passes but either specificity interval fails','DEFECT-SPECIFIC POSITIVE TRANSFER if primary material_positive and both specificity tests pass and neither main robustness sensitivity has major contrary reversal','NO MATERIAL TRANSFER as conservative overall fallback if primary-specific gain is materially contradicted by prespecified robustness; explicitly retain the positive primary result and explain veto; never invent fifth category'],'family':'same materiality, one confirmatory family only; no broad cross-family claims; confirmatory family tested separately, no pooling or multiplicity-adjusted global claim','Go_interpretation':'GO for further separately authorized research only if category A; category B partial nonspecific support; C null material-transfer result; D negative-transfer result; no automatic next phase'},
'stopping_rules':['run every prespecified seed/model/analysis regardless of outcome','no performance-driven retuning, threshold changes, exclusion or model substitution','stop on source hash mismatch, leakage or nonfinite predictions; fix implementation with transparent log without changing scientific plan','checkpoint completed deterministic jobs; resume failures only, no repeated success fits','no SHAP/mechanism/screening/secondDFT/manuscript'],
'runtime':{'parallel_workers':6,'per_model_threads':2,'deterministic_cache_equivalence':True}}
js('analysis_plan_phase3b.json',plan);js('preregistration_status.json',dict(plan_sha256=sha(OUT/'analysis_plan_phase3b.json'),models_fitted=0,prior_files_verified=len(snapshot),configurations=len(configs),unique_auxiliary_jobs=len(jobs)))
print('Frozen plan:',sha(OUT/'analysis_plan_phase3b.json'),'configs',len(configs),'auxiliary jobs',len(jobs),'prior files',len(snapshot))
