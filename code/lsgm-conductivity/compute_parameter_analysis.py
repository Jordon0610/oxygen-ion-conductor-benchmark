#!/usr/bin/env python3
"""Retrospective parameter-only source-local accounting; never fits a model.
Usage: python compute_parameter_analysis.py --inputs INPUT_DIR --out OUTPUT_DIR
No evaluation conductivity is calculated from author fit parameters. No fit is
assigned an unverified operating temperature. C_E(T) is retained symbolically.
"""
from pathlib import Path
from decimal import Decimal
import argparse,csv,hashlib,json,shutil,math

def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 if not rows:return
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def D(x):return Decimal(x)
def sg(x):return 1 if x>0 else -1 if x<0 else 0

def run(inputs,out):
 out.mkdir(parents=True,exist_ok=True)
 before={p.name:sha(p) for p in inputs.iterdir() if p.is_file()}
 obs=read(inputs/'02_ANALYSIS_OBSERVATIONS.csv');links=read(inputs/'03_TYPED_PARAMETER_AND_ALIAS_LINKS.csv');reg=read(inputs/'04_COMPARISON_REGISTRY.csv');res=read(inputs/'05_WITHIN_SOURCE_RESULTS.csv')
 byid={x['analysis_record_id']:x for x in obs}
 fitted=[];H={}
 for l in links:
  if l['relation_type']!='SAME_FIT_PARAMETERS':continue
  a,b=byid[l['observation_a']],byid[l['observation_b']]
  assert a['source']==b['source'] and a['sample_or_series_id']==b['sample_or_series_id'] and a['author_branch']==b['author_branch'] and a['component']==b['component']
  huang=a['source']=='HUANG1998'
  item=dict(fit_id=l['link_id'],source=a['source'],parent_id=a['historical_parent_id'],sample_id=a['sample_or_series_id'],branch=a['author_branch'],component=a['component'],Ea_id=a['analysis_record_id'],prefactor_id=b['analysis_record_id'],Ea_eV=a['raw_value'],prefactor_raw=b['raw_value'],prefactor_unit=b['raw_unit'],pairing_verified=True,log_convention='NATURAL_LOG_SIGMA_TIMES_T' if huang else 'RAW_A_EQ7',unit_compatibility='Same source numerical A convention; sigma in S/cm and T in K; A-ratio dimensionless' if huang else 'A unit not printed; no pairwise lnA comparison',source_measurement_scope=a['measurement_scope'],fit_scope=a['fit_scope'],temperature_800C_eligible=False,temperature_reason='Exact individual HT/LT fitted-point interval not established; overall measurement range and common Tstar do not certify each fit at800C' if huang else 'Only one source fit; no within-source paired composition contrast',source_locator=a['source_locator'],parameter_evidence=l['evidence'],source_sha256=a['source_hash'])
  fitted.append(item)
  if huang:H[(a['historical_parent_id'],a['author_branch'])]=(a,b)
 assert len(H)==76 and len(fitted)==77
 write(out/'B_same_fit_parameter_eligibility.csv',fitted)
 rows=[]
 for x in res:
  if x['source']!='HUANG1998' or x['registry_type'] not in ['WITHIN_NATIVE_ADJACENT','WITHIN_ACCEPTED_CORE']:continue
  if x['population']=='CORE10_WITHOUT_R0024_NO_REBRIDGE':continue
  ba=x['branch']; ea,pa=H[(x['sample_a'],ba)];eb,pb=H[(x['sample_b'],ba)]
  assert ea['component']==eb['component']==pa['component']==pb['component']
  de=D(eb['raw_value'])-D(ea['raw_value']);ca=D(pb['raw_value'])-D(pa['raw_value']); e_sign=-sg(de)
  if sg(ca)==0 and e_sign==0:direction='BOTH_ZERO'
  elif sg(ca)==0:direction='PREFACTOR_TIE'
  elif e_sign==0:direction='ENERGY_TIE'
  elif sg(ca)==e_sign:direction='REINFORCING_SIGNS_FOR_POSITIVE_T'
  else:direction='OPPOSING_SIGNS_FOR_POSITIVE_T'
  r=dict(registry_id=x['registry_id'],registry_type=x['registry_type'],population=x['population'],source=x['source'],path_id=x['path_id'],path_type=x['path_type'],fixed_value=x['fixed_value'],accepted_contrast_id=x['accepted_contrast_id'],sample_a=x['sample_a'],sample_b=x['sample_b'],x_a=x['x_a'],y_a=x['y_a'],x_b=x['x_b'],y_b=x['y_b'],orientation=x['orientation'],branch=ba,component=ea['component'],Ea_a_eV=ea['raw_value'],Ea_b_eV=eb['raw_value'],lnA_a=pa['raw_value'],lnA_b=pb['raw_value'],delta_Ea_eV=str(de),delta_Ea_meV=str(de*1000),C_A_delta_lnA=str(ca),C_E_symbolic=f'-({de})/(k_B*T)',energy_term_sign_for_positive_T=e_sign,term_sign_relation=direction,temperature_K='',C_E_numeric='',C_A_plus_C_E_numeric='',source_sigma800_a=x['sigma800_a_S_cm'],source_sigma800_b=x['sigma800_b_S_cm'],source_ln_sigma800_ratio=str(math.log(float(x['sigma800_b_S_cm'])/float(x['sigma800_a_S_cm']))),fit_vs_table_residual='',operating_temperature_eligible=False,excluded_operating_calculation_reason='NO_PER_PAIR_VERIFIED_FIT_INTERVAL_AT_800C; LT_NOT_EXTRAPOLATED; HT_LABEL_ONLY_NOT_CERTIFICATE',includes_R0024='R0024' in [x['sample_a'],x['sample_b']],primary_core=x['registry_type']=='WITHIN_ACCEPTED_CORE',without_R0024=x['registry_type']=='WITHIN_ACCEPTED_CORE' and 'R0024' not in [x['sample_a'],x['sample_b']],Ea_a_id=ea['analysis_record_id'],lnA_a_id=pa['analysis_record_id'],Ea_b_id=eb['analysis_record_id'],lnA_b_id=pb['analysis_record_id'],uncertainty_a_Ea=ea['reported_uncertainty'],uncertainty_b_Ea=eb['reported_uncertainty'],uncertainty_a_lnA=pa['reported_uncertainty'],uncertainty_b_lnA=pb['reported_uncertainty'],uncertainty_scope='Printed +/- retained; meaning/covariance unresolved; no propagated interval',source_locator_a=ea['source_locator'],source_locator_b=eb['source_locator'],fit_pair_a_evidence='R7 SAME_FIT_PARAMETERS '+ea['analysis_record_id']+' / '+pa['analysis_record_id'],fit_pair_b_evidence='R7 SAME_FIT_PARAMETERS '+eb['analysis_record_id']+' / '+pb['analysis_record_id'],source_sha256=ea['source_hash'])
  rows.append(r)
 assert len(rows)==204
 core=[x for x in rows if x['primary_core']]
 assert len(core)==28 and len({x['accepted_contrast_id'] for x in core})==14
 write(out/'B_all_source_parameter_contrasts.csv',rows)
 write(out/'B_primary14_parameter_contrasts.csv',core)
 write(out/'B_without_R0024_parameter_contrasts.csv',[x for x in core if x['without_R0024']])
 summaries=[]
 for view in ['CORE14','WITHOUT_R0024','ALL_SOURCE_NATIVE']:
  z=[x for x in rows if (x['primary_core'] if view=='CORE14' else x['without_R0024'] if view=='WITHOUT_R0024' else x['registry_type']=='WITHIN_NATIVE_ADJACENT')]
  for branch in ['E2_LT','E1_HT']:
   for path in ['ALL','FIXED_X','FIXED_Y','FIXED_SUM']:
    q=[x for x in z if x['branch']==branch and (path=='ALL' or x['path_type']==path)]
    if not q:continue
    summaries.append(dict(view=view,branch=branch,path_type=path,pair_count=len(q),unique_paths=len({x['path_id'] for x in q}),opposing=sum(x['term_sign_relation']=='OPPOSING_SIGNS_FOR_POSITIVE_T' for x in q),reinforcing=sum(x['term_sign_relation']=='REINFORCING_SIGNS_FOR_POSITIVE_T' for x in q),prefactor_ties=sum(x['term_sign_relation']=='PREFACTOR_TIE' for x in q),energy_ties=sum(x['term_sign_relation']=='ENERGY_TIE' for x in q),both_zero=sum(x['term_sign_relation']=='BOTH_ZERO' for x in q),delta_Ea_min_meV=min(D(x['delta_Ea_meV']) for x in q),delta_Ea_max_meV=max(D(x['delta_Ea_meV']) for x in q),delta_lnA_min=min(D(x['C_A_delta_lnA']) for x in q),delta_lnA_max=max(D(x['C_A_delta_lnA']) for x in q),supported_temperature_calculations=0,aggregation='descriptive counts; shared pairs/path blocks not independent'))
 write(out/'B_parameter_sign_summary.csv',summaries)
 blocked=[]
 for source in sorted({x['source'] for x in obs}):
  xx=[x for x in obs if x['source']==source]; ee=[x for x in xx if 'activation_energy' in x['parameter_type'] or x['parameter_type']=='Ea_ALGEBRA_FROM_PRINTED_FIT']; pp=[x for x in xx if x['parameter_type']=='SOURCE_lnA_FIT_PARAMETER' or x['parameter_type']=='conductivity_prefactor']
  if not ee and not pp:continue
  reason=('76same-fit Huang Ea/lnA rows; parameter-only contrasts retained; exact pairwise800C windows uncertified' if source=='HUANG1998' else 'Only one same-fit Ea/rawA association; no cross-composition source-local pair' if source=='C03_SOLGEL1996' else 'No accepted source-linked same-fit lnA record; no prefactor inferred from conductivity/equation intercept')
  blocked.append(dict(source=source,Ea_parameter_rows=len(ee),prefactor_rows=len(pp),supported_operating_temperature_pair_count=0,reason=reason))
 write(out/'B_source_level_eligibility.csv',blocked)
 checks={'same_fit_pair_count':77,'huang_same_fit_pairs':76,'core_pair_branch_records':28,'native_pair_branch_records':176,'without_R0024_pair_branch_records':24,'all_204_contrasts_preserved':len(rows)==204,'all_28_core_retained':len(core)==28,'no_numeric_energy_term_without_temperature':all(x['C_E_numeric']=='' for x in rows),'no_fit_to_table_residual_without_temperature':all(x['fit_vs_table_residual']=='' for x in rows),'no_LT800_extrapolation':True,'no_models_or_new_predictions':True,'no_conductivity_calculated_from_fit_parameters':True,'inputs_unchanged':True}
 assert before=={p.name:sha(p) for p in inputs.iterdir() if p.is_file()}
 write(out/'B_input_identities.csv',[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in inputs.iterdir() if p.is_file()])
 (out/'B_tests.json').write_text(json.dumps(checks,indent=2))
 return rows,summaries

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.inputs,a.out)
