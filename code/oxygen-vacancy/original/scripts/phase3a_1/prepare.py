"""Published-label ingestion, source identity checks and stateless composition features. No fitting."""
from pathlib import Path
import sys,json,hashlib,tarfile,re,warnings
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/phase3a_1';sys.path.insert(0,str(ROOT/'scripts'))
from audit_core import normalize_formula
EXPECTED_SHA='12f2f270127f98be9f1dc5e73619f1c995ab21b71f3a94493081d5890f37e596'
TARGETS=['Ef_min_sourcefiltered_q0','Ef_mean_sourcefiltered_q0']
FAMILIES=['Apatite','Perovskite-related','Scheelite','δ-Bi2O3']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def chem(formula):
 from pymatgen.core import Composition
 n=normalize_formula(formula)
 if n['normalization_status']!='confident_numeric_nominal':return ''
 return '-'.join(sorted(e for e,amount in Composition(n['normalized_formula']).get_el_amt_dict().items() if e!='O' and amount>0))
def main():
 source=ROOT/'data/phase3a/raw/charge0.csv';assert sha(source)==EXPECTED_SHA,'STOP: published CSV SHA mismatch'
 pub=pd.read_csv(source,float_precision='round_trip');assert len(pub)==1745 and pub.formula.nunique()==824
 assert not pub.full_name.duplicated().any();assert pub.vacancy_formation_energy.notna().all()
 # Structural identity and quality flags only. No raw value is assigned to a supervised label.
 parents=[];raw_details={}
 for path in sorted((ROOT/'data/phase3a/raw/archives').glob('*.tar.gz')):
  with tarfile.open(path) as t:
   def read(suffix):return json.load(t.extractfile(next(m for m in t.getmembers() if m.name.endswith('/'+suffix) and '/._' not in m.name)))
   b=read('bulk_visual_data.json');d=read('defect_visual_data.json')
  st=b['structure_graph']['structure'];geom={'lattice':st['lattice']['matrix'],'sites':sorted([{'species':x['species'],'abc':x['abc']} for x in st['sites']],key=lambda x:json.dumps(x,sort_keys=True))}
  fingerprint=hashlib.sha256(json.dumps(geom,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  parents.append(dict(raw_formula=b['formula'],normalized_formula=normalize_formula(b['formula'])['normalized_formula'],host_id=b['mp_id'],structure_sha256=fingerprint,space_group=b['space_group'],archive=path.name))
  raw_details[b['formula']]=d
 parents=pd.DataFrame(parents);identities=[];provenance=[]
 for formula,q in pub.groupby('formula',sort=True):
  n=normalize_formula(formula);assert n['normalization_status']=='confident_numeric_nominal';matches=parents[parents.normalized_formula.eq(n['normalized_formula'])]
  ids=matches.host_id.unique();geometries=matches.structure_sha256.unique()
  identity='unique_parent_host' if len(ids)==len(geometries)==1 else 'multiple_structural_hosts' if len(ids)>1 or len(geometries)>1 else 'unmapped'
  identities.append(dict(formula=formula,formula_id=n['normalized_formula'],normalized_formula=n['normalized_formula'],published_labels=len(q),parent_host_count=len(ids),parent_geometry_count=len(geometries),host_ids=';'.join(ids),space_groups=';'.join(matches.space_group.unique()),structure_sha256s=';'.join(geometries),identity_status=identity,phase_identity_claim='Published formula maps to source parent only; no experimental phase identity inferred.'))
  # A site label cannot resolve multiple parents; stop for curation rather than guessing a site-to-host mapping.
  assert identity=='unique_parent_host',f'Identity requires source curation: {formula}'
  parent=matches.iloc[0];raw=raw_details[parent.raw_formula]
  for r in q.itertuples():
   published_site=r.full_name[len(formula)+1:];raw_site=published_site
   if formula=='Sr2Zr7O16':raw_site={'Va_O3':'Va_O7','Va_O4':'Va_O8'}.get(published_site,published_site)
   entry=raw['defect_energy_summary']['defect_energies'][raw_site];i=entry['charges'].index(0);detail=entry['defect_energies'][i];full=raw_site+'_0'
   flags=raw.get('unusual_defects',{}).get(full,[]);state=raw['defect_states'][full]
   assert detail['is_shallow'] is False and state=='completed';assert formula not in ['K2Cd2O3','Ba2Ti(GeO4)2','Cs2O']
   assert abs(detail['formation_energy']-r.vacancy_formation_energy)<1e-12
   provenance.append(dict(published_label_id=r.full_name,csv_row_index=int(r.Index),formula=formula,formula_id=n['normalized_formula'],host_id=parent.host_id,
    published_site=published_site,raw_site=raw_site,charge_state=0,source_energy_eV=r.vacancy_formation_energy,
    source_status=state,is_shallow=False,source_unstable_host=False,raw_energy_agrees_after_documented_mapping=True,
    unusual_flags=json.dumps(flags),unusual_flagged=bool(flags),Sr_namespace_mapping_applied=published_site!=raw_site,
    source_label_population='official charge0.csv; energy never reconstructed from raw'))
 identity=pd.DataFrame(identities);pro=pd.DataFrame(provenance)
 identity.to_csv(OUT/'formula_host_identity_audit.csv',index=False)
 text=(ROOT/'data/source_audit/dft_source_quality_audit.md').read_text();section=text.split('### The 15 unusual-flagged neutral labels actually retained in B')[1].split('## 8.')[0]
 listed=set()
 for line in section.splitlines():
  cells=[v.strip() for v in line.strip().strip('|').split('|')]
  if len(cells)>=2 and cells[1].startswith('Va_O'):listed.add(cells[0]+'_'+cells[1].removesuffix('_0'))
 c1=set(pro.loc[pro.formula.eq('Sr2Zr7O16'),'published_label_id']);c2=set(pro.loc[pro.unusual_flagged,'published_label_id'])
 assert len(c1)==4 and len(c2)==15 and c2==listed
 exclusions={'B':set(),'C1':c1,'C2':c2,'C3':c1|c2}
 for population,excluded in exclusions.items():pro['in_'+population]=~pro.published_label_id.isin(excluded)
 pro.to_csv(OUT/'published_site_population_manifest.csv',index=False,float_format='%.17g')
 exclusion_rows=[];targetrows=[]
 for population,excluded in exclusions.items():
  for label in sorted(excluded):exclusion_rows.append(dict(population=population,published_label_id=label,reason='Sr2Zr7O16 namespace sensitivity' if label in c1 else 'retained source unusual-flag sensitivity'))
  selected=pro[pro['in_'+population]]
  for formula,q in selected.groupby('formula',sort=True):
   # Only values parsed from the pinned CSV populate these target aggregations.
   v=q.source_energy_eV.to_numpy();fid=q.formula_id.iloc[0]
   targetrows.append(dict(population=population,formula_id=fid,formula=formula,normalized_formula=fid,host_id=q.host_id.iloc[0],chemical_system=chem(formula),
    Ef_min_sourcefiltered_q0=float(v.min()),Ef_mean_sourcefiltered_q0=float(v.mean()),retained_site_count=len(v),
    median_eV=float(np.median(v)),maximum_eV=float(v.max()),std_population_eV=float(v.std(ddof=0)),published_label_ids=';'.join(q.published_label_id)))
 targets=pd.DataFrame(targetrows);targets.to_csv(OUT/'source_clean_host_targets.csv',index=False,float_format='%.17g')
 pd.DataFrame(exclusion_rows).to_csv(OUT/'source_quality_exclusion_manifest.csv',index=False)
 B=targets[targets.population.eq('B')].copy();assert B.normalized_formula.nunique()==824 and not B.host_id.duplicated().any()
 manifest=B[['formula_id','formula','normalized_formula','host_id','chemical_system']].copy();manifest['formulas_in_chemical_system']=manifest.groupby('chemical_system').formula_id.transform('size')
 manifest.to_csv(OUT/'chemical_system_manifest.csv',index=False)
 sizes=manifest.groupby('chemical_system').size().sort_values(ascending=False)
 summary={'published_SHA256':sha(source),'published_labels':len(pub),'published_formulas':pub.formula.nunique(),'unique_parent_formulas':int(identity.identity_status.eq('unique_parent_host').sum()),
 'chemical_systems':len(sizes),'formulas_per_system_distribution':{str(k):int(v) for k,v in sizes.value_counts().sort_index().items()},'largest_systems':{k:int(v) for k,v in sizes.head(15).items()},
 'populations':{pop:{'labels':int(pro['in_'+pop].sum()),'formulas':int(targets.population.eq(pop).sum()),'chemical_systems':int(targets.loc[targets.population.eq(pop),'chemical_system'].nunique())} for pop in exclusions},
 'C1_C2_intersection':len(c1&c2)}
 js(OUT/'source_population_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
 # Exactly the Phase-3A stateless feature recipe and versions; no globally fitted preprocessing.
 from pymatgen.core import Composition
 from matminer.featurizers.composition import ElementProperty,Stoichiometry
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');mag=ElementProperty.from_preset('magpie',impute_nan=False);st=Stoichiometry(p_list=[0,2,3,5,7,10],num_atoms=False)
 columns=mag.feature_labels()+st.feature_labels()+['oxygen_atomic_fraction'];values=[]
 for row in B.itertuples():
  comp=Composition(row.formula);values.append(list(mag.featurize(comp))+list(st.featurize(comp))+[comp.get_atomic_fraction('O')])
 X=pd.DataFrame(values,index=pd.Index(B.formula_id,name='formula_id'),columns=columns).replace([np.inf,-np.inf],np.nan)
 old=pd.read_csv(ROOT/'results/phase3a/dft_features.csv',index_col='host_id',float_precision='round_trip').loc[B.host_id]
 assert list(old.columns)==columns and np.allclose(X.to_numpy(),old.to_numpy(),equal_nan=True,rtol=1e-12,atol=1e-12)
 X.to_csv(OUT/'composition_features.csv',float_format='%.17g')
 (OUT/'feature_dictionary.csv').write_bytes((ROOT/'results/phase3a/feature_dictionary.csv').read_bytes())
 cols=['row_id','Formula','normalized_formula','normalization_status','family','final_component','cohort_K','connected_group_id']
 meta=pd.read_csv(ROOT/'results/benchmark/cohort_manifest.csv',usecols=cols);meta=meta[meta.cohort_K];assert len(meta)==313
 meta.to_csv(OUT/'experimental_metadata_only.csv',index=False)
 E=pd.read_csv(ROOT/'results/phase3a/experimental_composition_features.csv',index_col='row_id',float_precision='round_trip');E.loc[meta.row_id].to_csv(OUT/'experimental_composition_features.csv',float_format='%.17g')
 # Pretraining removal manifests are outcome-free feasibility bookkeeping, not actual pretraining.
 clean=[];removals=[]
 for fam in FAMILIES:
  e=meta[meta.family.eq(fam)];keys=set(e.normalized_formula);systems={chem(f) for f in e.Formula};systems.discard('')
  exact=B.normalized_formula.isin(keys);cs=B.chemical_system.isin(systems)
  assert not (exact & ~cs).any()
  for row in B.itertuples():removals.append(dict(family=fam,formula_id=row.formula_id,formula=row.formula,host_id=row.host_id,chemical_system=row.chemical_system,remove_exact_composition=row.normalized_formula in keys,remove_chemical_system=row.chemical_system in systems))
  clean.append(dict(family=fam,experimental_K_rows=len(e),test_chemical_systems=len(systems),exact_composition_removals=int(exact.sum()),remaining_DFT_formulas_exact=int((~exact).sum()),
   remaining_DFT_systems_exact=B.loc[~exact,'chemical_system'].nunique(),chemical_system_removals=int(cs.sum()),remaining_DFT_formulas_system=int((~cs).sum()),remaining_DFT_systems_system=B.loc[~cs,'chemical_system'].nunique(),
   fraction_DFT_formulas_remaining_system=float((~cs).mean()),does_not_collapse_by_registered_minimum=bool((~cs).sum()>=100 and B.loc[~cs,'chemical_system'].nunique()>=20)))
 pd.DataFrame(clean).to_csv(OUT/'clean_pretraining_feasibility.csv',index=False);pd.DataFrame(removals).to_csv(OUT/'clean_pretraining_removal_manifest.csv',index=False)
if __name__=='__main__':main()
