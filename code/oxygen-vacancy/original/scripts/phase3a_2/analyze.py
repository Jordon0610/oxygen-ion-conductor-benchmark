"""Saved-prediction calibration and descriptor-only transfer preflight; no estimators."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np,pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from pymatgen.core import Composition
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'results/phase3a_1';OUT=ROOT/'results/phase3a_2'
MODELS=['Ridge','RBF-SVR','ExtraTreesRegressor','XGBoost'];TARGET='Ef_min_sourcefiltered_q0'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csv(name,df):df.to_csv(OUT/name,index=False)
def elements(f):return {e for e,a in Composition(f).get_el_amt_dict().items() if a>0}
def system(f):return '-'.join(sorted(elements(f)-{'O'}))
class DomainReference:
 def __init__(self,X,H,scope):
  self.ids=list(X.index);self.columns=list(X.columns);self.scope=scope
  a=X.to_numpy(float);self.median=np.array([np.nanmedian(a[:,j]) if np.isfinite(a[:,j]).any() else 0. for j in range(a.shape[1])])
  a=np.where(np.isfinite(a),a,self.median);self.mean=a.mean(0);self.std=a.std(0);self.mask=self.std>1e-3*np.maximum(1.,np.abs(self.mean))
  assert self.mask.any();self.z=self.transform(X);self.H=H.loc[self.ids];self.elements=set().union(*(elements(f) for f in self.H.formula))
  dd=cdist(self.z,self.z)/np.sqrt(self.mask.sum());keys=self.H.normalized_formula.to_numpy();dd[keys[:,None]==keys[None,:]]=np.inf
  self.reference=dd.min(1);self.threshold=float(np.quantile(self.reference,.95));assert np.isfinite(self.threshold)
 def transform(self,X):
  assert list(X.columns)==self.columns
  a=np.where(np.isfinite(X.to_numpy(float)),X.to_numpy(float),self.median)
  return ((a-self.mean)/np.where(self.std>0,self.std,1))[:,self.mask]
 def evaluate(self,X,formulas):
  d=cdist(self.transform(X),self.z)/np.sqrt(self.mask.sum());rows=[]
  for i,idx in enumerate(X.index):
   missing=elements(formulas.loc[idx])-self.elements;j=int(d[i].argmin());value=float(d[i,j]);inside=value<=self.threshold
   rows.append(dict(record_id=idx,nearest_distance=value,threshold_95pct=self.threshold,distance_threshold_ratio=value/self.threshold,
    nearest_DFT_formula_id=self.ids[j],within_distance_threshold=bool(inside),all_elements_represented=not missing,unseen_elements=';'.join(sorted(missing)),in_domain=bool(inside and not missing)))
  return pd.DataFrame(rows).set_index('record_id')
 def record(self):
  return dict(scope=self.scope,fit_formula_ids=self.ids,fit_rows=len(self.ids),features_kept=int(self.mask.sum()),retained_features=np.array(self.columns)[self.mask].tolist(),
   medians=self.median.tolist(),means=self.mean.tolist(),std=self.std.tolist(),threshold_95pct=self.threshold,reference_nearest_distances=self.reference.tolist(),reference_elements=sorted(self.elements))
def ci(a):
 a=np.asarray(a);a=a[np.isfinite(a)];return np.quantile(a,[.025,.975]).tolist() if len(a) else [np.nan,np.nan]
def calibration(X,H,refs):
 splits=json.loads((OLD/'splits.json').read_text())['B']['S2'];dist=[]
 for s in splits:
  assert set(s['train']).isdisjoint(s['test'])
  assert set(H.loc[s['train'],'chemical_system']).isdisjoint(H.loc[s['test'],'chemical_system'])
  ref=DomainReference(X.loc[s['train']],H,f"S2_fold{s['fold']}");refs.append(ref.record())
  q=ref.evaluate(X.loc[s['test']],H.formula);q.index.name='formula_id';q['fold']=s['fold'];q['chemical_system']=H.loc[q.index,'chemical_system'];dist.append(q.reset_index())
 d=pd.concat(dist).sort_values('formula_id');assert len(d)==824 and d.formula_id.is_unique
 # Descriptor-only quintile edges, fixed probability levels before inspecting any losses.
 d['distance_quintile']=pd.qcut(d.nearest_distance,5,labels=[f'Q{i}' for i in range(1,6)])
 edges=np.quantile(d.nearest_distance,[0,.2,.4,.6,.8,1]);js('distance_quintile_edges.json',dict(probabilities=[0,.2,.4,.6,.8,1],edges=edges.tolist(),descriptor_only=True))
 csv('dft_oof_domain_distances.csv',d)
 p=pd.read_csv(OLD/'predictions.csv',float_precision='round_trip');p=p[p.population.eq('B')&p.protocol.eq('S2')&p.target.eq(TARGET)&p.model.isin(MODELS)]
 p=p.merge(d,on=['formula_id','chemical_system','fold'],validate='many_to_one');assert len(p)==4*824
 assert np.allclose(p.absolute_error_eV,np.abs(p.observed_eV-p.predicted_eV),rtol=1e-14,atol=1e-14)
 csv('dft_oof_error_distance_pairs.csv',p)
 groups=sorted(d.chemical_system.unique());gidx=pd.Categorical(d.chemical_system,categories=groups).codes
 rng=np.random.default_rng(424242);counts=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=5000)
 positions=[np.flatnonzero(gidx==j) for j in range(len(groups))]
 # Same whole-system resamples used for every statistic/model. Rank Spearman after resampling.
 samples=[np.concatenate([np.tile(positions[j],n) for j,n in enumerate(c) if n]) for c in counts]
 summary=[]
 for model in MODELS:
  q=p[p.model.eq(model)].set_index('formula_id').loc[d.formula_id];err=q.absolute_error_eV.to_numpy();x=q.nearest_distance.to_numpy()
  rho=float(spearmanr(x,err).statistic);rb=np.array([spearmanr(x[s],err[s]).statistic for s in samples]);lo,hi=ci(rb)
  summary.append(dict(model=model,statistic='Spearman',domain_rule='raw_distance',stratum='all',N_formulas=len(q),N_chemical_systems=len(groups),estimate=rho,CI_low=lo,CI_high=hi,unit='correlation',valid_bootstrap_draws=int(np.isfinite(rb).sum())))
  def bootmean(mask):
   sums=np.bincount(gidx,weights=err*mask,minlength=len(groups));ns=np.bincount(gidx,weights=mask.astype(float),minlength=len(groups));den=counts@ns
   return np.divide(counts@sums,den,out=np.full(len(counts),np.nan),where=den>0)
  def recordmean(mask,rule,stratum):
   boot=bootmean(mask);lo,hi=ci(boot)
   summary.append(dict(model=model,statistic='MAE',domain_rule=rule,stratum=stratum,N_formulas=int(mask.sum()),N_chemical_systems=len(set(gidx[mask])),estimate=float(err[mask].mean()) if mask.any() else np.nan,CI_low=lo,CI_high=hi,unit='eV',valid_bootstrap_draws=int(np.isfinite(boot).sum())))
   return boot
  for qu in ['Q1','Q2','Q3','Q4','Q5']:recordmean(q.distance_quintile.eq(qu).to_numpy(),'distance_quintiles',qu)
  for rule,col in [('distance_only','within_distance_threshold'),('distance_and_elements','in_domain')]:
   mask=q[col].to_numpy(bool);bi=recordmean(mask,rule,'in_domain');bo=recordmean(~mask,rule,'out_of_domain');gap=bo-bi;lo,hi=ci(gap)
   rel=1-err[mask].mean()/err[~mask].mean();summary.append(dict(model=model,statistic='out_minus_in_MAE',domain_rule=rule,stratum='contrast',N_formulas=len(q),N_chemical_systems=len(groups),estimate=float(err[~mask].mean()-err[mask].mean()),CI_low=lo,CI_high=hi,unit='eV',valid_bootstrap_draws=int(np.isfinite(gap).sum()),relative_MAE_reduction=float(rel),materially_informative=bool(rel>=.1 and lo>0)))
   lo,hi=ci(1-bi/bo);summary.append(dict(model=model,statistic='relative_MAE_reduction',domain_rule=rule,stratum='contrast',N_formulas=len(q),N_chemical_systems=len(groups),estimate=float(rel),CI_low=lo,CI_high=hi,unit='fraction',valid_bootstrap_draws=int(np.isfinite(1-bi/bo).sum())))
  print('Calibrated',model,flush=True)
 result=pd.DataFrame(summary);csv('dft_domain_calibration.csv',result);return result

def preflight(X,H,refs):
 cols=['row_id','Formula','family','normalized_formula','normalization_status','connected_group_id','final_component','cohort_K','normalized_doi']
 full=pd.read_csv(ROOT/'results/benchmark/cohort_manifest.csv',usecols=cols).set_index('row_id');meta=full[full.cohort_K].copy();assert len(meta)==312
 assert full.normalization_status.eq('confident_numeric_nominal').all()
 full['chemical_system']=full.Formula.map(system)
 E=pd.read_csv(OLD/'experimental_composition_features.csv',index_col='row_id',float_precision='round_trip').loc[meta.index];assert list(E.columns)==list(X.columns)
 oldmeta=pd.read_csv(OLD/'experimental_metadata_only.csv').set_index('row_id');assert (meta.connected_group_id==oldmeta.loc[meta.index,'connected_group_id']).all()
 splits=[s for s in json.loads((ROOT/'results/benchmark/splits.json').read_text()) if s['cohort']=='K_FE' and s['protocol'] in ['P3','P4']]
 coverage=[];rowpairs=[];removals=[];closure=[]
 for s in splits:
  tg=set(meta.loc[s['test_ids'],'connected_group_id']);train=meta.loc[s['train_ids']];test=meta.loc[s['test_ids']]
  assert tg.isdisjoint(train.connected_group_id);close=full[full.connected_group_id.isin(tg)]
  for rid,r in close.iterrows():closure.append(dict(split_id=s['split_id'],row_id=rid,original_K_test_row=rid in s['test_ids'],normalized_formula=r.normalized_formula,chemical_system=r.chemical_system,connected_group_id=r.connected_group_id))
  for method,key in [('exact_clean','normalized_formula'),('system_clean','chemical_system')]:
   removed=H.index[H[key].isin(set(close[key]))].tolist();allowed=H.index[~H.index.isin(removed)].tolist();assert set(allowed).isdisjoint(removed)
   for fid in H.index:removals.append(dict(split_id=s['split_id'],cleaning=method,formula_id=fid,chemical_system=H.loc[fid,'chemical_system'],removed=fid in removed))
   ref=DomainReference(X.loc[allowed],H,s['split_id']+'/'+method);rec=ref.record();rec.update(excluded_formula_ids=removed,test_group_ids=sorted(tg));refs.append(rec)
   q=ref.evaluate(E,meta.Formula);q=q.join(meta[['family','connected_group_id','final_component','normalized_formula']]);q.index.name='row_id'
   roles={rid:role for role,ids in [('train',s['train_ids']),('test',s['test_ids']),('purged',s['purged_ids'])] for rid in ids}
   q['role']=[roles[i] for i in q.index];q['split_id']=s['split_id'];q['protocol']=s['protocol'];q['repeat']=s['repeat'];q['fold']=s['fold'];q['cleaning']=method;rowpairs.append(q.reset_index())
   tr=q[q.role.eq('train')];te=q[q.role.eq('test')];ct=te[te.in_domain];cr=tr[tr.in_domain]
   coverage.append(dict(split_id=s['split_id'],protocol=s['protocol'],repeat=s['repeat'],fold=s['fold'],family=s['held_out_family'],cleaning=method,
    original_train_rows=len(tr),original_train_groups=tr.connected_group_id.nunique(),original_test_rows=len(te),original_test_groups=te.connected_group_id.nunique(),
    test_group_closure_rows=len(close),closure_extra_rows=len(close)-len(te),removed_DFT_formulas=len(removed),remaining_DFT_formulas=len(allowed),remaining_DFT_chemical_systems=H.loc[allowed,'chemical_system'].nunique(),
    reference_threshold_95pct=ref.threshold,retained_features=int(ref.mask.sum()),covered_train_rows=len(cr),covered_train_groups=cr.connected_group_id.nunique(),covered_test_rows=len(ct),covered_test_groups=ct.connected_group_id.nunique(),
    bulk_covered_rows=int(ct.final_component.eq('bulk').sum()),bulk_covered_groups=ct[ct.final_component.eq('bulk')].connected_group_id.nunique(),
    total_component_covered_rows=int(ct.final_component.eq('total').sum()),total_component_covered_groups=ct[ct.final_component.eq('total')].connected_group_id.nunique(),
    test_groups_at_least_3=bool(ct.connected_group_id.nunique()>=3),family_classification=('confirmatory' if ct.connected_group_id.nunique()>=5 else 'exploratory' if ct.connected_group_id.nunique()>=3 else 'case-study only') if s['protocol']=='P4' else 'not a family-OOD experiment'))
  print('Preflight',s['split_id'],flush=True)
 c=pd.DataFrame(coverage);pairs=pd.concat(rowpairs,ignore_index=True);csv('experimental_row_fold_eligibility.csv',pairs);csv('dft_pretraining_exclusion_manifest.csv',pd.DataFrame(removals));csv('experimental_test_group_closure.csv',pd.DataFrame(closure))
 for method,name in [('exact_clean','fold_specific_exact_clean_coverage.csv'),('system_clean','fold_specific_system_clean_coverage.csv')]:csv(name,c[c.protocol.eq('P3')&c.cleaning.eq(method)])
 csv('p4_family_transfer_preflight.csv',c[c.protocol.eq('P4')])
 aggregates=[];components=[]
 for method,pp in pairs[pairs.protocol.eq('P3')&pairs.role.eq('test')].groupby('cleaning'):
  eligible=pp[pp.in_domain]
  scopes=[('union_across_repeats',eligible.drop_duplicates('row_id'))]
  for repeat,g in eligible.groupby('repeat'):scopes.append((f'repeat_{repeat}',g))
  counts=eligible.groupby('row_id')['repeat'].nunique();common=set(counts[counts==2].index);scopes.append(('eligible_in_both_repeats',eligible[eligible.row_id.isin(common)].drop_duplicates('row_id')))
  for scope,g in scopes:
   aggregates.append(dict(cleaning=method,scope=scope,eligible_unique_rows=g.row_id.nunique(),eligible_unique_global_groups=g.connected_group_id.nunique()))
   for component in ['bulk','total']:
    z=g[g.final_component.eq(component)];components.append(dict(cleaning=method,scope=scope,component=component,covered_rows=z.row_id.nunique(),covered_groups=z.connected_group_id.nunique(),meets_aggregate_25_group_requirement=bool(z.connected_group_id.nunique()>=25)))
  for split_id,g in eligible.groupby('split_id'):
   for component in ['bulk','total']:
    z=g[g.final_component.eq(component)];components.append(dict(cleaning=method,scope=split_id,component=component,covered_rows=len(z),covered_groups=z.connected_group_id.nunique(),meets_aggregate_25_group_requirement=False))
 for _,r in c[c.protocol.eq('P4')].iterrows():
  for comp,prefix in [('bulk','bulk'),('total','total_component')]:components.append(dict(cleaning=r.cleaning,scope=r.split_id,component=comp,covered_rows=r[prefix+'_covered_rows'],covered_groups=r[prefix+'_covered_groups'],meets_family_5_group_requirement=bool(r[prefix+'_covered_groups']>=5)))
 a=pd.DataFrame(aggregates);csv('aggregate_eligible_population_summary.csv',a);csv('component_transfer_preflight.csv',pd.DataFrame(components));return c,a

def run():
 planhash=sha(OUT/'analysis_plan_phase3a_2.json');status=json.loads((OUT/'preregistration_status.json').read_text());assert planhash==status['plan_sha256']
 assert not (OUT/'calculation_started.json').exists(),'No silent re-execution'
 for f,h in json.loads((OUT/'analysis_plan_phase3a_2.json').read_text())['inputs'].items():assert sha(ROOT/f)==h,f
 js('calculation_started.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=planhash))
 H=pd.read_csv(OLD/'chemical_system_manifest.csv').set_index('formula_id');X=pd.read_csv(OLD/'composition_features.csv',index_col='formula_id',float_precision='round_trip').loc[H.index]
 assert len(H)==824 and H.normalized_formula.is_unique and (H.chemical_system==H.formula.map(system)).all()
 refs=[];cal=calibration(X,H,refs);cov,agg=preflight(X,H,refs);js('domain_reference_trace.json',refs)
 saved=pd.read_csv(OLD/'source_clean_dft_model_results.csv');s=saved[saved.target.eq(TARGET)&saved.protocol.eq('S2')&saved.model.isin(MODELS)]
 assert s.sort_values('MAE_eV').iloc[0].model=='ExtraTreesRegressor'
 exact=cov[cov.protocol.eq('P3')&cov.cleaning.eq('exact_clean')];groups=int(agg[agg.cleaning.eq('exact_clean')&agg.scope.eq('union_across_repeats')].eligible_unique_global_groups.iloc[0]);minfold=int(exact.covered_test_groups.min())
 learned=bool(s.supported.sum()>=3 and s[s.model.eq('ExtraTreesRegressor')].supported.iloc[0]);inform=bool(cal[cal.model.eq('ExtraTreesRegressor')&cal.statistic.eq('out_minus_in_MAE')&cal.domain_rule.eq('distance_and_elements')].materially_informative.iloc[0])
 hard=groups>=25 and minfold>=3 and learned;families=cov[cov.protocol.eq('P4')&cov.cleaning.eq('exact_clean')&cov.covered_test_groups.ge(5)].family.tolist()
 gate=dict(exact_clean_unique_global_groups=groups,minimum_fold_eligible_test_groups=minfold,frozen_strong_learning_and_primary_pass=learned,primary_joint_calibration_materially_informative=inform,hard_aggregate_conditions_pass=hard,aggregate_decision='NO-GO' if not hard else 'GO' if inform else 'CONDITIONAL GO',individually_confirmatory_exact_clean_families=families,broad_confirmatory_cross_family_feasible=len(families)>=2,experimental_models_fitted=0,DFT_models_fitted=0)
 js('decision_gate.json',gate);assert sha(OUT/'analysis_plan_phase3a_2.json')==planhash
 js('calculation_completed.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=planhash,reference_fits=len(refs),predictive_estimator_fits=0))
 print(json.dumps(gate,indent=2));print(agg.to_string(index=False));print(cov[cov.protocol.eq('P4')][['family','cleaning','covered_test_rows','covered_test_groups','bulk_covered_groups','total_component_covered_groups']].to_string(index=False))
if __name__=='__main__':run()
