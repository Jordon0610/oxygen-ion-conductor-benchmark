"""Paired chemical-system uncertainty and DFT-only source-clean domain reference."""
import numpy as np,pandas as pd,json
from scipy.spatial.distance import cdist
from pymatgen.core import Composition
from prepare import OUT,ROOT,TARGETS,FAMILIES,js
from engine import FoldPreprocessor

def bootstrap_arrays(q,seed=424242):
 systems=sorted(q.chemical_system.unique());rng=np.random.default_rng(seed)
 counts=rng.multinomial(len(systems),np.ones(len(systems))/len(systems),size=5000)
 sizes=q.groupby('chemical_system').size().reindex(systems).to_numpy();denom=counts@sizes
 return systems,counts,denom

def metrics():
 p=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');records=[]
 for (population,target,protocol),q in p.groupby(['population','target','protocol'],sort=False):
  base=q[q.model.eq('DummyRegressor')].set_index('formula_id').sort_index();systems,counts,denom=bootstrap_arrays(base)
  be=base.absolute_error_eV;bs=base.groupby('chemical_system').absolute_error_eV.sum().reindex(systems).to_numpy();bootbase=counts@bs/denom
  for model,g in q.groupby('model',sort=False):
   g=g.set_index('formula_id').loc[base.index];assert np.array_equal(g.observed_eV,base.observed_eV)
   errors=g.absolute_error_eV;mae=float(errors.mean());baseline=float(be.mean());rs=g.groupby('chemical_system').absolute_error_eV.sum().reindex(systems).to_numpy()
   boot=counts@rs/denom;gain=bootbase-boot;relative=1-boot/bootbase
   ci=np.quantile(boot,[.025,.975]);gi=np.quantile(gain,[.025,.975]);ri=np.quantile(relative,[.025,.975]);obs=g.observed_eV.to_numpy()
   r2=1-g.squared_error_eV.sum()/np.sum((obs-obs.mean())**2) if len(obs)>=20 and np.std(obs)>=.1 else np.nan
   records.append(dict(population=population,target=target,protocol=protocol,model=model,N_formulas=len(g),N_chemical_systems=len(systems),MAE_eV=mae,
    MAE_CI_low_eV=ci[0],MAE_CI_high_eV=ci[1],RMSE_eV=float(np.sqrt(g.squared_error_eV.mean())),R2=r2,Dummy_MAE_eV=baseline,
    Dummy_relative_improvement=1-mae/baseline,relative_improvement_CI_low=ri[0],relative_improvement_CI_high=ri[1],
    paired_Dummy_minus_model_eV=baseline-mae,paired_gain_CI_low_eV=gi[0],paired_gain_CI_high_eV=gi[1],
    positive_improvement_direction=bool(mae<baseline),supported=bool(model!='DummyRegressor' and 1-mae/baseline>=.10 and ri[0]>0)))
 result=pd.DataFrame(records);result[result.population.eq('B')].to_csv(OUT/'source_clean_dft_model_results.csv',index=False,float_format='%.12g')
 result[result.protocol.eq('S2')].to_csv(OUT/'source_quality_sensitivity_results.csv',index=False,float_format='%.12g')
 contrasts=[]
 for (target,model),q in p[p.population.eq('B')].groupby(['target','model'],sort=False):
  a=q[q.protocol.eq('S1')].set_index('formula_id').sort_index();b=q[q.protocol.eq('S2')].set_index('formula_id').loc[a.index]
  delta=b.absolute_error_eV-a.absolute_error_eV;systems,counts,denom=bootstrap_arrays(a)
  vals=pd.DataFrame({'system':a.chemical_system,'delta':delta}).groupby('system').delta.sum().reindex(systems).to_numpy();ci=np.quantile(counts@vals/denom,[.025,.975])
  contrasts.append(dict(target=target,model=model,S1_MAE_eV=a.absolute_error_eV.mean(),S2_MAE_eV=b.absolute_error_eV.mean(),S2_minus_S1_MAE_eV=delta.mean(),paired_system_CI_low_eV=ci[0],paired_system_CI_high_eV=ci[1]))
 pd.DataFrame(contrasts).to_csv(OUT/'validation_protocol_contrasts.csv',index=False)
 support=[]
 for (pop,target,protocol),q in result.groupby(['population','target','protocol'],sort=False):
  count=int(q.supported.sum());support.append(dict(population=pop,target=target,protocol=protocol,passing_models=count,support='strong' if count>=3 else 'partial' if count==2 else 'weak',positive_direction_models=int(q[q.model.ne('DummyRegressor')].positive_improvement_direction.sum())))
 pd.DataFrame(support).to_csv(OUT/'support_summary.csv',index=False)
 print(result[['population','target','protocol','model','MAE_eV','Dummy_relative_improvement','supported']].to_string(index=False),flush=True)

def domain():
 H=pd.read_csv(OUT/'source_clean_host_targets.csv');h=H[H.population.eq('B')].set_index('formula_id')
 X=pd.read_csv(OUT/'composition_features.csv',index_col='formula_id',float_precision='round_trip').loc[h.index]
 meta=pd.read_csv(OUT/'experimental_metadata_only.csv').set_index('row_id');E=pd.read_csv(OUT/'experimental_composition_features.csv',index_col='row_id',float_precision='round_trip').loc[meta.index]
 # The domain audit is the explicit exception allowing a full B-only reference. No experimental row fits it.
 prep=FoldPreprocessor(True).fit(X);z=prep.transform(X);ez=prep.transform(E);nf=z.shape[1]
 dd=cdist(z,z)/np.sqrt(nf);keys=h.normalized_formula.to_numpy();dd[keys[:,None]==keys[None,:]]=np.inf
 reference=dd.min(axis=1);threshold=float(np.quantile(reference,.95));dist=cdist(ez,z)/np.sqrt(nf)
 elements=set().union(*(set(Composition(f).get_el_amt_dict()) for f in h.formula));rows=[]
 for i,(rowid,r) in enumerate(meta.iterrows()):
  resolved=r.normalization_status=='confident_numeric_nominal';j=int(dist[i].argmin());n=h.iloc[j]
  unseen=set(Composition(r.Formula).get_el_amt_dict())-elements if resolved else set();value=float(dist[i,j]) if resolved else np.nan
  exact=h[h.normalized_formula.eq(r.normalized_formula)]
  rows.append(dict(row_id=rowid,formula=r.Formula,normalized_formula=r.normalized_formula,family=r.family,conductivity_component=r.final_component,connected_group_id=r.connected_group_id,
   nearest_DFT_formula_id=n.name if resolved else '',nearest_DFT_host=n.host_id if resolved else '',nearest_DFT_formula=n.formula if resolved else '',nearest_distance=value,
   B_reference_threshold_95pct=threshold,within_distance_threshold=bool(resolved and value<=threshold),all_elements_represented=bool(resolved and not unseen),
   unseen_elements=';'.join(sorted(unseen)),in_domain=bool(resolved and value<=threshold and not unseen),exact_normalized_composition_exists=len(exact)>0,
   exact_source_host_ids=';'.join(exact.host_id),polymorph_caution='No experimental structure identity established; composition equality is not phase identity.'))
 d=pd.DataFrame(rows);d.to_csv(OUT/'source_clean_experimental_domain_distance.csv',index=False)
 # Row-level manifest preserves every original K row, including uncovered records.
 d[['row_id','family','connected_group_id','in_domain']].to_csv(OUT/'covered_experimental_group_manifest.csv',index=False)
 summaries=[]
 for fam in ['all K']+FAMILIES:
  q=d if fam=='all K' else d[d.family.eq(fam)];covered=q[q.in_domain];unique=q.drop_duplicates('normalized_formula')
  orig_groups=q.connected_group_id.nunique();ng=covered.connected_group_id.nunique()
  summaries.append(dict(domain=fam,original_rows=len(q),original_groups=orig_groups,in_domain_rows=len(covered),in_domain_groups=ng,
   fraction_original_rows=len(covered)/len(q),fraction_original_groups=ng/orig_groups,original_compositions=len(unique),
   in_domain_compositions=int(unique.in_domain.sum()),composition_coverage=unique.in_domain.mean(),distance_only_rows=int(q.within_distance_threshold.sum()),
   elements_represented_rows=int(q.all_elements_represented.sum()),retains_G_ge_5=bool(fam in FAMILIES and ng>=5)))
 summary=pd.DataFrame(summaries);summary.to_csv(OUT/'covered_experimental_group_summary.csv',index=False)
 js(OUT/'domain_preprocessing.json',dict(fit_formula_ids=list(X.index),experimental_fit_rows=0,source_population='B',features_kept=int(prep.mask.sum()),retained_features=list(X.columns[prep.mask]),
  medians=prep.median.tolist(),means=prep.mean.tolist(),standard_deviations=prep.std.tolist(),threshold_95pct=threshold,reference_elements=sorted(elements)))
 pd.DataFrame({'formula_id':h.index,'nearest_other_composition_distance':reference}).to_csv(OUT/'B_reference_distances.csv',index=False)
 print(summary.to_string(index=False),flush=True)

def gate():
 s=pd.read_csv(OUT/'support_summary.csv');d=pd.read_csv(OUT/'covered_experimental_group_summary.csv')
 count=int(s[s.population.eq('B')&s.protocol.eq('S2')&s.target.eq(TARGETS[0])].passing_models.iloc[0]);ng=int(d[d.domain.eq('all K')].in_domain_groups.iloc[0]);nf=int(d.retains_G_ge_5.sum())
 decision='NO-GO' if count<=1 or ng<25 or nf==0 else 'GO' if count>=3 and nf>=2 else 'CONDITIONAL GO'
 js(OUT/'decision_gate.json',dict(primary_S2_passing_models=count,primary_S2_support='strong' if count>=3 else 'partial' if count==2 else 'weak',
  covered_K_rows=int(d[d.domain.eq('all K')].in_domain_rows.iloc[0]),covered_K_global_groups=ng,qualifying_family_count=nf,qualifying_families=d.loc[d.retains_G_ge_5,'domain'].tolist(),
  aggregate_domain_requirement_passes=ng>=25,broad_family_domain_requirement_passes=nf>=2,NO_GO_precedence_applied=True,decision=decision,experimental_transfer_executed=False))
if __name__=='__main__':metrics();domain();gate()
