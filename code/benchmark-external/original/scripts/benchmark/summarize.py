"""Evaluate saved outer predictions only after the frozen fitting run finishes."""
import json
import numpy as np
import pandas as pd
from .design import OUT,MODELS
from .engine import assert_frozen


def cluster_draws(d,stratified=False):
    """Return resampling multiplicities per original row; paired columns reuse these draws."""
    rng=np.random.default_rng(8675309);weights=np.zeros((2000,len(d)))
    strata=d.groupby('family',sort=True).indices if stratified else {'all':np.arange(len(d))}
    # No group may be split into independently resampled strata.
    if stratified: assert d.groupby('connected_group_id').family.nunique().max()==1
    for ix in strata.values():
        keys=d.iloc[ix].connected_group_id.to_numpy();groups=np.unique(keys)
        if len(groups)<5: return None
        counts=rng.multinomial(len(groups),np.repeat(1/len(groups),len(groups)),size=2000)
        lookup={g:j for j,g in enumerate(groups)}
        weights[:,ix]=counts[:,[lookup[g] for g in keys]]
    return weights


def intervals(d,columns,stratified=False,macro=False):
    w=cluster_draws(d,stratified)
    if w is None: return {c:np.full(2000,np.nan) for c in columns}
    values=d[columns].to_numpy()
    if macro:
        estimates=[]
        for ix in d.groupby('family',sort=True).indices.values():
            estimates.append((w[:,ix]@values[ix])/w[:,ix].sum(axis=1)[:,None])
        a=np.mean(estimates,axis=0)
    else: a=(w@values)/w.sum(axis=1)[:,None]
    return {c:a[:,i] for i,c in enumerate(columns)}


def bounds(a):
    return tuple(np.quantile(a,[.025,.975])) if np.isfinite(a).all() else (np.nan,np.nan)


def aggregate(d,macro=False,stratified=False):
    n=len(d);g=d.connected_group_id.nunique();f=d.family.nunique()
    avg=d.groupby('family')[['ae','se','dummy_ae']].mean().mean() if macro else d[['ae','se','dummy_ae']].mean()
    b=intervals(d,['ae','dummy_ae'],stratified,macro)
    lo,hi=bounds(b['ae']);ilo,ihi=bounds(1-b['ae']/b['dummy_ae'])
    sd=d.y_true.std(ddof=1);sst=((d.y_true-d.y_true.mean())**2).sum()
    r2ok=n>=20 and g>=5 and sd>=100 and sst>0 and not macro
    reason='' if r2ok else ('macro-family R2 not defined' if macro else 'requires N>=20, G>=5, target SD>=100 meV and nonzero SST')
    min_g=d.groupby('family').connected_group_id.nunique().min() if stratified else g
    return dict(N_rows=n,N_groups=g,N_families=f,MAE_meV=avg.ae,MAE_CI_low=lo,MAE_CI_high=hi,
        RMSE_meV=np.sqrt(avg.se),Dummy_MAE_meV=avg.dummy_ae,baseline_relative_improvement=1-avg.ae/avg.dummy_ae,
        improvement_CI_low=ilo,improvement_CI_high=ihi,R2=1-d.se.sum()/sst if r2ok else np.nan,
        R2_omission_reason=reason,target_SD_meV=sd,
        uncertainty_note='unavailable: fewer than 5 groups' if min_g<5 else 'fragile: at least one stratum has 5-9 groups' if min_g<10 else 'conditional cluster interval; no refitting',
        aggregation='macro_family' if macro else 'row_weighted')


def load_predictions():
    manifest=pd.read_csv(OUT/'cohort_manifest.csv').set_index('row_id')
    splits=json.loads((OUT/'splits.json').read_text());records=[]
    for s in splits:
        checkpoints={m:json.loads((OUT/'checkpoints'/(s['split_id']+'__'+m+'.json')).read_text()) for m in MODELS}
        dummy=checkpoints['DummyRegressor']['predictions']
        for model,p in checkpoints.items():
            for rid,pred,base in zip(p['test_ids'],p['predictions'],dummy):
                r=manifest.loc[rid];y=float(r.Ea)
                records.append(dict(split_id=s['split_id'],cohort=s['cohort'],protocol=s['protocol'],repeat=s['repeat'],
                    model=model,row_id=rid,family=r.family,connected_group_id=r.connected_group_id,component=r.final_component,
                    y_true=y,y_pred=pred,dummy_pred=base,ae=abs(pred-y),se=(pred-y)**2,dummy_ae=abs(base-y),
                    training_rows=p['train_rows'],training_groups=p['train_groups'],other_family_rows_purged=p['purged_rows']))
    return pd.DataFrame(records)


def per_row(p):
    keys=['cohort','protocol','model','row_id','family','connected_group_id','component','y_true']
    return p.groupby(keys,sort=False,dropna=False)[['ae','se','dummy_ae','training_rows','training_groups','other_family_rows_purged']].mean().reset_index()


def paired(a,b,label,stratified=False):
    keys=['row_id','family','connected_group_id','y_true']
    d=a[keys+['ae']].merge(b[['row_id','ae']],on='row_id',suffixes=('_new','_reference'),validate='one_to_one')
    d['delta']=d.ae_new-d.ae_reference
    v=intervals(d,['delta'],stratified)['delta'];lo,hi=bounds(v)
    delta=d.delta.mean();ref=d.ae_reference.mean()
    return dict(comparison=label,N_rows=len(d),N_groups=d.connected_group_id.nunique(),
        reference_MAE_meV=ref,new_MAE_meV=d.ae_new.mean(),delta_MAE_meV=delta,delta_CI_low=lo,delta_CI_high=hi,
        relative_increase=delta/ref,material_point_increase=bool(delta>=25 and delta>=.1*ref),
        supported_material_increase=bool(delta>=25 and delta>=.1*ref and lo>0))


def main():
    assert_frozen();p=load_predictions();p.to_csv(OUT/'outer_predictions.csv',index=False)
    d=per_row(p);d.to_csv(OUT/'per_record_losses.csv',index=False)
    coverage=pd.read_csv(OUT/'confirmatory_family_manifest.csv')
    result=[];family=[]
    for (cohort,protocol,model),x in d.groupby(['cohort','protocol','model'],sort=False):
        eligible=set(coverage.loc[(coverage.cohort==cohort)&coverage.P4_eligible,'family'])
        for scope,z in [('all_tested',x),('matched_confirmatory',x[x.family.isin(eligible)])]:
            if z.empty: continue
            for macro in ([False,True] if protocol=='P4' else [False]):
                result.append(dict(cohort=cohort,protocol=protocol,model=model,scope=scope,
                    **aggregate(z,macro,protocol=='P4'),mean_training_rows=z.training_rows.mean(),
                    mean_training_groups=z.training_groups.mean()))
        if protocol=='P4':
            for f,z in x.groupby('family'):
                family.append(dict(cohort=cohort,model=model,family=f,**aggregate(z,False,True),
                    training_rows=int(z.training_rows.iloc[0]),training_groups=int(z.training_groups.iloc[0]),
                    other_family_rows_purged=int(z.other_family_rows_purged.iloc[0]),status='success'))
    pd.DataFrame(result).to_csv(OUT/'model_results.csv',index=False)
    pd.DataFrame(family).to_csv(OUT/'family_ood_results.csv',index=False)
    comparisons=[];sens=[];quality=[]
    for model in MODELS:
        for cohort in ['K','B','T','R']:
            c=d[(d.model==model)&(d.cohort==cohort)]
            for new,ref in [('P2','P1'),('P3','P2'),('P3','P1')]:
                row=dict(cohort=cohort,model=model,**paired(c[c.protocol==new],c[c.protocol==ref],new+' minus '+ref))
                comparisons.append(row)
                if cohort in ['K','B','T'] and new=='P3' and ref=='P1': sens.append(dict(analysis='within-component protocol shift',**row))
            a=c[c.protocol=='P4'];b=c[c.protocol=='P3']
            if not a.empty:
                row=dict(cohort=cohort,model=model,**paired(a,b,'P4 minus P3, matched family rows',True))
                comparisons.append(row)
                if cohort in ['K','B','T']: sens.append(dict(analysis='within-component family shift',**row))
        for cohort,comp in [('B','bulk'),('T','total')]:
            for protocol in ['P3','P4']:
                a=d[(d.model==model)&(d.cohort==cohort)&(d.protocol==protocol)]
                b=d[(d.model==model)&(d.cohort=='K')&(d.protocol==protocol)&(d.component==comp)]
                sens.append(dict(analysis='component-only training vs K, identical evaluation rows',cohort=cohort,model=model,
                    **paired(a,b,protocol+': '+cohort+' minus K on shared '+comp+' records',protocol=='P4')))
        for protocol in ['P3','P4']:
            a=d[(d.model==model)&(d.cohort=='KQ')&(d.protocol==protocol)]
            b=d[(d.model==model)&(d.cohort=='K')&(d.protocol==protocol)]
            quality.append(dict(model=model,protocol=protocol,**paired(a,b,'KQ minus K on shared quality-eligible rows',protocol=='P4')))
    pd.DataFrame(comparisons).to_csv(OUT/'protocol_comparisons.csv',index=False)
    pd.DataFrame(sens).to_csv(OUT/'component_sensitivity.csv',index=False)
    pd.DataFrame(quality).to_csv(OUT/'quality_sensitivity.csv',index=False)
    # Expose conventional-reference overlap rather than calling P1/P2 provenance-safe.
    rows=pd.read_csv(OUT/'cohort_manifest.csv').set_index('row_id');leaks=[]
    for s in json.loads((OUT/'splits.json').read_text()):
        tr=rows.loc[s['train_ids']];te=rows.loc[s['test_ids']]
        leaks.append(dict(split_id=s['split_id'],cohort=s['cohort'],protocol=s['protocol'],test_rows=len(te),
            test_rows_with_training_DOI=int(te.normalized_doi.isin(tr.normalized_doi).sum()),
            test_rows_with_training_composition=int(te.normalized_formula.isin(tr.normalized_formula).sum()),
            test_rows_with_training_connected_group=int(te.connected_group_id.isin(tr.connected_group_id).sum())))
    pd.DataFrame(leaks).to_csv(OUT/'observed_split_overlap.csv',index=False)
    assert_frozen();print('All prespecified metrics and paired contrasts saved.')


if __name__=='__main__': main()
