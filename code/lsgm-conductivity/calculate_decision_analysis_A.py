#!/usr/bin/env python3
"""New retrospective descriptive material-choice analysis, never model fitting.
Run with --inputs packaged_input_directory --output new_output_directory.
All evaluated conductivities are directly transcribed source-table values.
No Ea-to-conductivity conversion occurs. No model objects or sklearn imports.
"""
from pathlib import Path
from decimal import Decimal
import argparse, csv, hashlib, json, math, re
from collections import defaultdict

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows,fields=None):
    rows=list(rows);p.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def number(s):
    s=str(s).strip().replace('−','-').replace(' ','')
    if not s:return None
    try:return Decimal(s)
    except Exception:
        m=re.fullmatch(r'([+\-0-9.]+)[×x]10\^([+\-0-9]+)',s)
        if m:return Decimal(m[1])*(Decimal(10)**int(m[2]))
        raise ValueError(f'Not a literal source number: {s}')
def choose(va,vb,a,b,smaller=True):
    if va==vb:return ''
    return a if ((va<vb) if smaller else (va>vb)) else b

def loss(selected,a,b,sa,sb):
    best=max(sa,sb)
    ra=math.log10(float(best/sa));rb=math.log10(float(best/sb))
    if selected:
        r=ra if selected==a else rb
        ratio=float((sa if selected==a else sb)/best)
        return r,ratio,r,ratio
    # A tie remains an unresolved directional choice. These expectations are
    # labelled separately and are not used to infer a favourable selection.
    return '', '',(ra+rb)/2,float((sa+sb)/(2*best))

def main(inp,out):
    out.mkdir(parents=True,exist_ok=True)
    initial={p.name:sha(p) for p in inp.iterdir() if p.is_file()}
    core=read(inp/'R7__03_core14_target_context.csv')
    core=[r for r in core if r['branch']=='E2_LT']
    assert len(core)==14 and len({r['path_id'] for r in core})==10
    assert len({r[k] for r in core for k in ['sample_a','sample_b']})==11
    local=read(inp/'R9__Figure4_all_matched_parent_predictions.csv')
    local=[r for r in local if r['arm']=='FE_MATCHED']
    glob=read(inp/'R9_global_FE_P3_LSGM_filtered_saved_predictions.csv')
    gd={(r['row_id'],r['repeat'],r['model']):r for r in glob}
    assert len(gd)==len(glob)==228 and len(local)==228
    binds=[]
    for l in local:
        g=gd[(l['row_id'],l['repeat'],l['model'])]
        assert g['split_id']==f"R9_K_FE_P3_r{l['repeat']}_f2"
        assert Decimal(g['y_true'])==Decimal(l['observed_E2_meV_original'])
        assert g['connected_group_id']==l['connected_group_id']=='G0001'
        assert g['component']==l['component']=='bulk'
        assert g['training_rows']==l['training_rows']=={'0':'189','1':'233'}[l['repeat']]
        assert g['training_groups']==l['training_groups']=='38'
        d=abs(Decimal(g['y_pred'])-Decimal(l['prediction_meV']))
        # Display precision only; no scientific comparison tolerance is tuned.
        assert d < Decimal('1e-9')
        binds.append(dict(row_id=l['row_id'],repeat=l['repeat'],model=l['model'],
                          local_split_id=l['split_id'],global_split_id=g['split_id'],
                          local_saved_prediction_meV=l['prediction_meV'],global_saved_prediction_meV=g['y_pred'],
                          printed_decimal_difference_meV=str(d),target_meV=g['y_true'],component=g['component'],
                          connected_group_id=g['connected_group_id'],training_rows=g['training_rows'],training_groups=g['training_groups'],
                          outcome='IDENTITY_BOUND_SAVED_PREDICTIONS;GLOBAL_R9_VALUE_USED'))
    write(out/'saved_prediction_identity_binding.csv',binds)
    write(out/'full_38_parent_prediction_context.csv',local)
    # Retain source-bound metadata/values in every decision row.
    common=['source','population','path_id','path_type','pair_id','accepted_contrast_id','sample_a','sample_b','x_a','y_a','x_b','y_b','delta_x','delta_y','delta_sum','orientation','component','branch','Ea_a_eV','Ea_b_eV','sigma800_a_id','sigma800_b_id','sigma800_a_S_cm','sigma800_b_S_cm','phase_a','phase_b','source_locators_a','source_locators_b','carrier_limit_a','carrier_limit_b','Ea_to_800C_scope','ordering_uncertainty']
    models=['ExtraTreesRegressor','XGBoost','DummyRegressor']
    decisions=[];paired=[]
    for i,c in enumerate(core,1):
        a,b=c['sample_a'],c['sample_b'];sa,sb=number(c['sigma800_a_S_cm']),number(c['sigma800_b_S_cm'])
        assert sa>0 and sb>0
        ea,eb=number(c['Ea_a_eV']),number(c['Ea_b_eV'])
        e=choose(ea,eb,a,b)
        ora=choose(sa,sb,a,b,smaller=False)
        for rep in ['0','1']:
            for model in models:
                ga,gb=gd[(a,rep,model)],gd[(b,rep,model)]
                pa,pb=Decimal(ga['y_pred']),Decimal(gb['y_pred'])
                assert Decimal(ga['y_true'])==ea*1000 and Decimal(gb['y_true'])==eb*1000
                m=choose(pa,pb,a,b)
                rr={}
                for rule,sel in [('M_predicted_Ea',m),('E_author_Ea',e),('O_source_conductivity',ora)]:
                    unique,ratio,exp,expratio=loss(sel,a,b,sa,sb)
                    row={k:c[k] for k in common}
                    row.update(dict(pair_number=i,repeat=rep,model=model,rule=rule,
                                    saved_prediction_a_meV=str(pa),saved_prediction_b_meV=str(pb),selected_parent=sel,
                                    unique_decision=bool(sel),unresolved_tie=not bool(sel),regret_decades_unique=unique,
                                    selected_to_best_sigma_ratio_unique=ratio,
                                    expected_regret_decades_equal_tie_choice=exp,
                                    expected_selected_to_best_sigma_ratio_equal_tie_choice=expratio,
                                    without_R0024=('R0024' not in [a,b]),
                                    statistical_scope='One source; shared pairs; descriptive central values; no CI or p-value',
                                    numerical_source='SAVED_R9_GLOBAL_FE_P3;NO_NEW_PREDICTIONS'))
                    decisions.append(row);rr[rule]=row
                rm=rr['M_predicted_Ea'];re_=rr['E_author_Ea']
                paired.append(dict(pair_number=i,contrast_id=c['accepted_contrast_id'],block_id=c['path_id'],
                                   parent_a=a,parent_b=b,repeat=rep,model=model,
                                   M_selection=m,E_selection=e,both_unique=bool(m and e),
                                   strict_disagreement_if_both_unique=(m!=e) if m and e else '',
                                   any_unresolved_tie=not bool(m and e),
                                   M_minus_E_expected_regret_decades=rm['expected_regret_decades_equal_tie_choice']-re_['expected_regret_decades_equal_tie_choice'],
                                   M_expected_regret_decades=rm['expected_regret_decades_equal_tie_choice'],
                                   E_expected_regret_decades=re_['expected_regret_decades_equal_tie_choice'],
                                   without_R0024=('R0024' not in [a,b]),
                                   interpretation='May be negative; prediction error can incidentally improve conductivity selection'))
    write(out/'all_pair_repeat_decisions.csv',decisions)
    write(out/'paired_M_versus_E_decisions.csv',paired)
    blocks=[];rep_summary=[];summary=[]
    for sub in ['PRIMARY_14_PAIRS_10_BLOCKS','WITHOUT_R0024_12_PAIRS_8_BLOCKS']:
        dd=[d for d in decisions if sub.startswith('PRIMARY') or d['without_R0024']]
        assert len({d['accepted_contrast_id'] for d in dd})==(14 if sub.startswith('PRIMARY') else 12)
        assert len({d['path_id'] for d in dd})==(10 if sub.startswith('PRIMARY') else 8)
        for model in models:
            for rule in ['M_predicted_Ea','E_author_Ea','O_source_conductivity']:
                for rep in ['0','1']:
                    ds=[d for d in dd if d['model']==model and d['rule']==rule and d['repeat']==rep]
                    bgs=defaultdict(list)
                    for d in ds:bgs[d['path_id']].append(d)
                    br=[]
                    for block,rows in bgs.items():
                        n=len(rows);v=sum(float(d['expected_regret_decades_equal_tie_choice']) for d in rows)/n
                        row=dict(subset=sub,model=model,rule=rule,repeat=rep,block_id=block,pairs=n,
                                 unique_decisions=sum(d['unique_decision'] for d in rows),
                                 unresolved_ties=sum(d['unresolved_tie'] for d in rows),
                                 decision_coverage=sum(d['unique_decision'] for d in rows)/n,
                                 mean_expected_regret_decades=v,
                                 mean_unique_decision_regret_decades=(sum(float(d['regret_decades_unique']) for d in rows if d['unique_decision'])/sum(d['unique_decision'] for d in rows)) if any(d['unique_decision'] for d in rows) else '',
                                 geometric_sigma_retained_ratio=10**(-v))
                        blocks.append(row);br.append(row)
                    v=sum(b['mean_expected_regret_decades'] for b in br)/len(br)
                    pp=[p for p in paired if p['model']==model and p['repeat']==rep and (sub.startswith('PRIMARY') or p['without_R0024'])]
                    ps=defaultdict(list)
                    for p in pp:ps[p['block_id']].append(p)
                    r=dict(subset=sub,model=model,rule=rule,repeat=rep,pairs=len(ds),blocks=len(br),
                           unique_decisions=sum(d['unique_decision'] for d in ds),unresolved_ties=sum(d['unresolved_tie'] for d in ds),
                           raw_pair_decision_coverage=sum(d['unique_decision'] for d in ds)/len(ds),
                           equal_block_decision_coverage=sum(b['decision_coverage'] for b in br)/len(br),
                           equal_block_expected_regret_decades=v,geometric_sigma_retained_ratio=10**(-v),
                           strict_M_E_disagreements_both_unique=sum(p['strict_disagreement_if_both_unique'] is True for p in pp),
                           M_E_unresolved_pairs=sum(p['any_unresolved_tie'] for p in pp),
                           equal_block_M_minus_E_regret_decades=sum(sum(p['M_minus_E_expected_regret_decades'] for p in q)/len(q) for q in ps.values())/len(ps))
                    rep_summary.append(r)
                rr=[r for r in rep_summary if r['subset']==sub and r['model']==model and r['rule']==rule]
                v=sum(r['equal_block_expected_regret_decades'] for r in rr)/2
                summary.append(dict(subset=sub,model=model,rule=rule,pairs_per_repeat=rr[0]['pairs'],blocks=rr[0]['blocks'],repeats=2,
                                    equal_block_repeat_mean_expected_regret_decades=v,geometric_sigma_retained_ratio=10**(-v),
                                    pair_repeat_unique_decisions=sum(r['unique_decisions'] for r in rr),
                                    pair_repeat_unresolved_ties=sum(r['unresolved_ties'] for r in rr),
                                    raw_pair_repeat_decision_coverage=sum(r['raw_pair_decision_coverage'] for r in rr)/2,
                                    equal_block_repeat_mean_decision_coverage=sum(r['equal_block_decision_coverage'] for r in rr)/2,
                                    paired_M_E_strict_disagreements=sum(r['strict_M_E_disagreements_both_unique'] for r in rr),
                                    paired_M_E_unresolved_pair_repeats=sum(r['M_E_unresolved_pairs'] for r in rr),
                                    equal_block_repeat_mean_M_minus_E_regret_decades=sum(r['equal_block_M_minus_E_regret_decades'] for r in rr)/2,
                                    weighting='pair mean within block, equal block mean within repeat, equal mean over repeats',
                                    tie_summary='equal-probability expected regret; unresolved choices are not directional successes',
                                    inference='descriptive only; one source, no independent-pair bootstrap'))
    write(out/'block_repeat_regret.csv',blocks)
    paired_blocks=[]
    for r in blocks:
        if r['rule']!='M_predicted_Ea':continue
        e=next(x for x in blocks if x['rule']=='E_author_Ea' and all(x[k]==r[k] for k in ['subset','model','repeat','block_id']))
        paired_blocks.append(dict(subset=r['subset'],model=r['model'],repeat=r['repeat'],block_id=r['block_id'],pairs=r['pairs'],M_mean_regret_decades=r['mean_expected_regret_decades'],E_mean_regret_decades=e['mean_expected_regret_decades'],M_minus_E_mean_regret_decades=r['mean_expected_regret_decades']-e['mean_expected_regret_decades']))
    write(out/'block_paired_M_versus_E_regret.csv',paired_blocks)
    write(out/'repeat_regret_summary.csv',rep_summary)
    write(out/'decision_regret_summary.csv',summary)
    # Optional native-adjacent author-Ea diagnostics kept by source and branch.
    native=read(inp/'R7__05_WITHIN_SOURCE_RESULTS.csv')
    nrows=[];nblocks=[];nsumm=[]
    for c in native:
        if c['source'] not in ['HP1996','STEVENSON1997'] or c['registry_type']!='WITHIN_NATIVE_ADJACENT':continue
        a,b=c['sample_a'],c['sample_b'];sa,sb=number(c['sigma800_a_S_cm']),number(c['sigma800_b_S_cm'])
        ea,eb=number(c['Ea_a_eV']),number(c['Ea_b_eV'])
        eligible=all(x is not None for x in [sa,sb,ea,eb]) and sa>0 and sb>0
        r={k:c.get(k,'') for k in common if k in c}
        r.update(dict(ML_evaluable=False,ML_non_evaluable_reason='No exact source-excluded saved prediction identity for these sample IDs; no model fitting authorized',
                      E_evaluable=eligible,temperature_use='Ea selection diagnostic against direct source sigma800; no extrapolation of Ea fit'))
        if eligible:
            e=choose(ea,eb,a,b);u,rat,exp,exprat=loss(e,a,b,sa,sb)
            _,_,uniform,_=loss('',a,b,sa,sb)
            r.update(dict(selected_parent=e,unique_decision=bool(e),unresolved_tie=not bool(e),regret_decades_unique=u,
                          selected_to_best_sigma_ratio_unique=rat,expected_E_regret_decades=exp,uninformative_expected_regret_decades=uniform,
                          E_minus_uninformative_regret_decades=exp-uniform))
        else:r['non_evaluable_reason']='Missing or nonpositive directly tabulated value or missing author Ea'
        nrows.append(r)
    write(out/'native_source_author_Ea_decisions.csv',nrows)
    for key in sorted({(r['source'],r['branch']) for r in nrows}):
        nr=[r for r in nrows if (r['source'],r['branch'])==key and r['E_evaluable']]
        bg=defaultdict(list)
        for r in nr:bg[r['path_id']].append(r)
        for block,rr in bg.items():
            nblocks.append(dict(source=key[0],branch=key[1],block_id=block,pairs=len(rr),
                                E_expected_regret_decades=sum(r['expected_E_regret_decades'] for r in rr)/len(rr),
                                uninformative_expected_regret_decades=sum(r['uninformative_expected_regret_decades'] for r in rr)/len(rr)))
        bs=[r for r in nblocks if (r['source'],r['branch'])==key]
        nsumm.append(dict(source=key[0],branch=key[1],pairs=len(nr),blocks=len(bs),unique_E_decisions=sum(r['unique_decision'] for r in nr),
                          E_expected_regret_decades_equal_block=sum(r['E_expected_regret_decades'] for r in bs)/len(bs),
                          uninformative_expected_regret_decades_equal_block=sum(r['uninformative_expected_regret_decades'] for r in bs)/len(bs),
                          component_scope=nr[0]['component'],inference='source/branch-specific descriptive only; overlapping pairs, no CI'))
    write(out/'native_source_block_regret.csv',nblocks)
    write(out/'native_source_regret_summary.csv',nsumm)
    tests={
      'source_hashes_unchanged':initial=={p.name:sha(p) for p in inp.iterdir() if p.is_file()},
      'saved_global_binding_rows':len(binds),'binding_max_printed_difference_meV':str(max(Decimal(b['printed_decimal_difference_meV']) for b in binds)),
      'all_38_parents_each_repeat_each_three_models':len(local)==38*2*3,
      'primary_11_parents_14_pairs_10_blocks':True,'without_R0024_12_pairs_8_blocks_no_rebridge':True,
      'both_frozen_repeats_retained':{r['repeat'] for r in decisions}=={'0','1'},
      'all_sigma_direct_positive':all(number(r['sigma800_a_S_cm'])>0 and number(r['sigma800_b_S_cm'])>0 for r in core),
      'exact_ties_retained_no_tie_break':all(r['selected_parent']=='' for r in decisions if r['unresolved_tie']),
      'dummy_all_pair_repeat_predictions_tied':all(r['unresolved_tie'] for r in decisions if r['model']=='DummyRegressor' and r['rule']=='M_predicted_Ea'),
      'oracle_expected_regret_zero':all(r['expected_regret_decades_equal_tie_choice']==0 for r in decisions if r['rule']=='O_source_conductivity'),
      'no_models_fitted_or_loaded':True,'no_new_predictions':True,'no_Ea_derived_evaluation_sigma':True,
      'no_independent_pair_inference':True,'method':'Retrospective descriptive; no confidence intervals or p-values added'}
    (out/'analysis_A_tests.json').write_text(json.dumps(tests,indent=2))
    (out/'input_hashes.json').write_text(json.dumps(initial,indent=2))
    return decisions,paired,summary,nrows,nsumm

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.inputs,a.output)
