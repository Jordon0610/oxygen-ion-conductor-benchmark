"""Freeze all scientific definitions and nested partitions before any estimator fit."""
import json,datetime,importlib.metadata as md
import pandas as pd
from sklearn.model_selection import GroupKFold
from prepare import ROOT,OUT,sha,js,TARGETS,EXPECTED_SHA

def main():
 p=OUT/'analysis_plan_phase3a_1.json';assert not p.exists() and not (OUT/'fit_started.json').exists()
 assert sha(ROOT/'data/phase3a/raw/charge0.csv')==EXPECTED_SHA
 H=pd.read_csv(OUT/'source_clean_host_targets.csv');B=H[H.population.eq('B')].sort_values('formula_id').reset_index(drop=True)
 splits={'B':{}};rows=[]
 for protocol,column in [('S1','normalized_formula'),('S2','chemical_system')]:
  groups=pd.factorize(B[column],sort=True)[0];folds=[]
  for fi,(tr,te) in enumerate(GroupKFold(5,shuffle=True,random_state=314159).split(B,groups=groups)):
   train=B.iloc[tr];test=B.iloc[te];inner=[];ig=pd.factorize(train[column],sort=True)[0]
   for ii,(it,iv) in enumerate(GroupKFold(3,shuffle=True,random_state=271828+fi).split(train,groups=ig)):
    inner.append(dict(train=train.iloc[it].formula_id.tolist(),test=train.iloc[iv].formula_id.tolist()))
   folds.append(dict(fold=fi,train=train.formula_id.tolist(),test=test.formula_id.tolist(),inner=inner))
  splits['B'][protocol]=folds
 for pop in ['C1','C2','C3']:
  allowed=set(H.loc[H.population.eq(pop),'formula_id']);folds=[]
  for original in splits['B']['S2']:
   inner=[{role:[x for x in item[role] if x in allowed] for role in ['train','test']} for item in original['inner']]
   folds.append(dict(fold=original['fold'],train=[x for x in original['train'] if x in allowed],test=[x for x in original['test'] if x in allowed],inner=inner))
  splits[pop]={'S2':folds}
 for pop,protocols in splits.items():
  h=H[H.population.eq(pop)].set_index('formula_id')
  for protocol,folds in protocols.items():
   for fold in folds:
    for role in ['train','test']:
     for fid in fold[role]:rows.append(dict(population=pop,protocol=protocol,outer_fold=fold['fold'],role=role,formula_id=fid,chemical_system=h.loc[fid,'chemical_system'],host_id=h.loc[fid,'host_id']))
 js(OUT/'splits.json',splits);pd.DataFrame(rows).to_csv(OUT/'split_manifest.csv',index=False)
 previous=json.loads((ROOT/'results/phase3a/analysis_plan_phase3a.json').read_text())
 inputs=[OUT/x for x in ['prior_phase_hashes.json','source_policy_hashes.json','source_clean_host_targets.csv','published_site_population_manifest.csv','source_quality_exclusion_manifest.csv','composition_features.csv','feature_dictionary.csv','experimental_metadata_only.csv','experimental_composition_features.csv','chemical_system_manifest.csv','formula_host_identity_audit.csv','split_manifest.csv','splits.json','transfer_target_policy.md','transfer_control_design.md']]
 inputs += [ROOT/'data/phase3a/raw/charge0.csv',ROOT/'data/source_audit/dft_source_quality_audit.md',ROOT/'data/source_audit/dft_source_quality_policy.csv',ROOT/'results/phase3a/analysis_plan_phase3a.json',ROOT/'results/benchmark/cohort_manifest.csv',ROOT/'scripts/audit_core.py',ROOT/'requirements-benchmark-lock.txt']
 inputs += list((ROOT/'scripts/phase3a_1').glob('*.py'))
 plan=dict(version=1,status='frozen before any Phase3A.1 estimator fit',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 source_population='B: pinned official charge0.csv only, 1745 source-filtered q=0 labels / 824 formulas. Raw export is provenance, never primary label reconstruction.',
 source_SHA256=EXPECTED_SHA,source_repository='kumagai-group/oxygen_vacancies_db',source_commit='e310353ecbf698499c85a5ec5416b76e82dcecfd',source_policy_hashes=json.loads((OUT/'source_policy_hashes.json').read_text()),
 frozen_input_hashes={str(x.relative_to(ROOT)):sha(x) for x in inputs},software_versions={x:md.version(x) for x in ['numpy','pandas','scikit-learn','xgboost','matminer','pymatgen']},
 target_order={'primary':TARGETS[0],'secondary':TARGETS[1]},target_unit='eV',
 target_definitions={'primary':'Minimum among retained published source-filtered neutral site labels of a verified formula/host; not universally lowest physical vacancy energy.',
 'secondary':'Unweighted arithmetic mean of retained published source-filtered neutral site labels; not thermodynamic, Boltzmann, multiplicity-weighted or equilibrium average.',
 'descriptive':'Retained count, median, maximum, population std ddof=0. No label transformations or imputed targets. Zero-retained-site formulas absent from that sensitivity target table only.'},
 host_identity='Independently map published normalized formula to raw parent MP identifier and exported structural fingerprint; all 824 uniquely mapped. Stop on ambiguous/multiple-parent site attribution rather than assume formula=structure. Formula equality never proves experimental phase identity.',
 chemical_system_definition='Exact rational normalized nominal formula; unordered set of elements with positive stoichiometric amount excluding O; lexicographically sorted element symbols joined by hyphen. No oxidation-state, phase, dopant or oxygen-content distinction within a system.',
 features={'definition':previous['features'],'pipeline':'Exactly Phase3A 139 composition features and software versions; independently recomputed and compared. Experimental descriptors reused only as stateless per-composition features.',
 'forbidden':['experimental Ea','EaHT','Phase2 errors','family outcomes','DOI','family labels as predictors','component as DFT predictor','test outcome information']},
 preprocessing=previous['preprocessing'],models=previous['models'],fixed_model_parameters=previous['fixed_model_parameters'],model_seed=previous['model_seed'],
 validation={'S1':'5 outer grouped folds by normalized formula; 3 inner grouped folds by normalized formula',
 'S2':'PRIMARY OOD: 5 outer non-oxygen chemical-system-disjoint folds; 3 inner chemical-system-disjoint folds',
 'algorithm':'GroupKFold shuffle=True; sorted formula IDs; group integers factorized in sorted order; outer random_state=314159; inner random_state=271828+fold',
 'sensitivity':'After all B results, C1/C2/C3 repeat S2 for both targets and all five models. Restrict B S2 outer and inner assignments to retained formulas; retune only within those inner training folds. Reaggregate targets from retained CSV labels.',
 'selection':'Same modest exhaustive grids as Phase3A; pooled inner-validation formula MAE; first candidate wins tie; no early stopping or outer-label selection.'},
 populations={'B':'official CSV membership and values unchanged','C1':'B minus four Sr2Zr7O16 labels','C2':'B minus 15 retained unusual-flagged published labels independently matched to source audit','C3':'B minus union of C1 and C2; sensitivity only'},
 metrics={'primary':'Pooled formula/host MAE eV','secondary':['RMSE eV','R2 only N>=20 and target std>=0.1 eV','Dummy-relative improvement'],
 'baseline':'Training-fold median, same outer training/test rows as predictive model',
 'uncertainty':'5000 paired chemical-system-cluster bootstrap draws, seed=424242. Resample G global chemical systems with replacement; all formula losses in a sampled system share its multiplicity. Preserve paired model/Dummy and S1/S2 losses. Pooled row-weighted MAE uses resampled total formula count. Percentile 95% bounds; no refitting during bootstrap.',
 'improvement':'Relative = 1 - model_MAE/Dummy_MAE; bootstrap ratio computed per draw. Also report paired absolute Dummy_MAE-model_MAE in eV.',
 'validation_contrast':'For B, paired S2-minus-S1 formula loss contrast with same chemical-system cluster resampling, all predictive models; no presumption of degradation.'},
 success_rule={'model_supported':'Relative point MAE improvement >=0.10 AND lower paired chemical-system-cluster 95% CI on relative improvement >0',
 'primary':'Ef_min_sourcefiltered_q0 under S2: strong >=3/4; partial 2/4; weak <=1/4 predictive models. Same calculation for Ef_mean is secondary only.',
 'sensitivity':'Report direction and support counts for every C population; never choose population by score.'},
 domain_coverage={'definition':{k:v for k,v in previous['domain_distance'].items() if k!='coverage_rule'},'reference_population':'All 824 B formulas only; refit reference/scaler from B. No source-raw-only host enters reference.',
 'distance':'RMS Euclidean distance in identical DFT-only imputed, near-constant-filtered standardized descriptor space',
 'threshold':'95th percentile of B leave-one-normalized-composition-out nearest-neighbor distances',
 'in_domain':'Distance <= threshold AND all constituent elements present in B; symbolic unresolved compositions not covered. No experimental label enters any decision.',
 'group_identity':'Reuse exact global connected_group_id from frozen Phase2 cohort_manifest. Do not regroup; count unique touching groups separately for K and each family.',
 'family_rule':'Only original four Phase2 confirmatory families assessed; in-domain confirmatory retention requires >=5 original global groups. No lower threshold.'},
 future_pretraining={'exact':'For each original K test family remove all exact normalized test-family compositions from B before any pretraining or preprocessing.',
 'chemical':'Sensitivity remove any B formula whose non-oxygen chemical system occurs in that entire original K family.',
 'practical_noncollapse_rule':'At least 100 remaining DFT formulas and 20 remaining chemical systems, reported with fractions. This is an operational feasibility minimum, not a power or performance guarantee.',
 'controls':'transfer_control_design.md frozen now; no experimental controls executed; target order never chosen from experimental scores.'},
 transfer_gate={'aggregate_requirement':'Covered K >=25 unique global experimental connected groups',
 'family_requirement':'At least two original confirmatory families each retain >=5 in-domain global groups',
 'precedence':'Explicit NO-GO conditions override overlapping CONDITIONAL-GO wording: covered groups <25 is NO-GO even if another condition passes.',
 'NO-GO':'Primary S2 support weak (<=1/4) OR covered K groups <25 OR zero qualifying families.',
 'GO':'Primary S2 support strong (>=3/4), covered K groups >=25 and >=2 qualifying families.',
 'CONDITIONAL GO':'Otherwise: partial S2 support (2/4), or exactly one qualifying family with >=25 covered K groups. With one family, broad confirmatory cross-family transfer is not supported; any family-specific transfer is exploratory.',
 'stopping':'No experimental transfer in this phase regardless of gate. If hash/count mismatch, source identity/label mismatch, leakage, nonfinite predictions or frozen-file change occurs, stop and report; do not repair by changing labels or selecting a better population.'})
 js(p,plan);js(OUT/'preregistration_status.json',dict(plan_sha256=sha(p),frozen_before_fit=True));print('FROZEN',sha(p))
if __name__=='__main__':main()
