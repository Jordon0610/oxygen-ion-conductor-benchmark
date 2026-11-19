"""Prespecified design, cohort ingestion, and outcome-independent partitions."""
import csv
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/benchmark'
sys.path.insert(0,str(ROOT/'scripts'))
from audit_core import normalize_formula, normalize_doi, connected_components, FAMILY_MAP

RAW_SHA='8ec66b131bae79f8520f7cf40e91e2be2541995939bc43b744eaac627f4fad34'
SEEDS=[1729,2718]
MODEL_SEED=31415
MODELS=['DummyRegressor','Ridge','RBF-SVR','ExtraTreesRegressor','XGBoost']
GRIDS={
 'DummyRegressor':[{}],
 'Ridge':[{'alpha':a} for a in [0.1,1.0,10.0,100.0]],
 'RBF-SVR':[{'C':c,'gamma':g,'epsilon':50.0} for c in [100.0,1000.0] for g in ['scale',0.01]],
 'ExtraTreesRegressor':[{'min_samples_leaf':l,'max_features':f} for l in [1,3] for f in [0.5,1.0]],
 'XGBoost':[{'max_depth':d,'learning_rate':lr} for d in [2,4] for lr in [0.05,0.1]],
}
FIXED={
 'DummyRegressor':{'strategy':'median'},
 'Ridge':{'fit_intercept':True},
 'RBF-SVR':{'kernel':'rbf'},
 'ExtraTreesRegressor':{'n_estimators':128,'max_depth':None,'bootstrap':False,'n_jobs':2,'random_state':MODEL_SEED},
 'XGBoost':{'n_estimators':150,'objective':'reg:squarederror','tree_method':'hist','subsample':1.0,
   'colsample_bytree':1.0,'reg_lambda':10.0,'n_jobs':2,'random_state':MODEL_SEED,'verbosity':0},
}
QUALITY_FLAGS={
 106:('different_permeation_endpoint','S04; exclusion_recommendation=separate_endpoint'),
 108:('severe_phase_regime_ambiguity','S05; oxygen-content crossover, impurity and temperature-range discrepancy'),
 208:('verified_multiphase','S09; delta + Er2O3'),
 209:('verified_multiphase','S09; delta + Er2O3'),
 210:('severe_phase_stability_ambiguity','S10; metastable low-temperature cubic state'),
 216:('verified_multiphase','S10; outside single-phase fcc interval'),
 267:('primary_Ea_regime_unresolved','S14; component_verified_but_fit_regime_review_required'),
 268:('primary_Ea_regime_unresolved','S14; component_verified_but_fit_regime_review_required'),
 269:('primary_Ea_regime_unresolved','S14; component_verified_but_fit_regime_review_required'),
 270:('primary_Ea_regime_unresolved','S14; component_verified_but_fit_regime_review_required'),
 280:('verified_multiphase','S15; Ca-series secondary-phase peaks'),
 283:('verified_multiphase','S15; Ba-series secondary-phase peaks'),
}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, rows, fields=None):
    OUT.mkdir(parents=True,exist_ok=True)
    if fields is None: fields=list(rows[0]) if rows else []
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def source_rows():
    rawpath=ROOT/'data/raw/oxygen_ion_conductor_dataset.csv'
    assert sha(rawpath)==RAW_SHA, 'Raw data changed'
    manifest=json.loads((ROOT/'data/curation/deliverable_hashes.json').read_text())
    for name,digest in manifest.items():
        assert sha(ROOT/'data/curation'/name)==digest, f'Curation source changed: {name}'
    with rawpath.open(newline='') as f: raw=list(csv.DictReader(f))
    with (ROOT/'data/curation/measurement_component_curation.csv').open(encoding='utf-8-sig',newline='') as f:
        curated=list(csv.DictReader(f))
    assert len(raw)==len(curated)
    rows=[]
    for i,(a,b) in enumerate(zip(raw,curated),1):
        assert int(b['dataset_row_id'])==i
        assert all(a[k]==b[k] for k in a), f'Original cell changed at record {i}'
        norm=normalize_formula(a['Formula'])
        rows.append(dict(b,**norm,row_id=f'R{i:04d}',normalized_doi=normalize_doi(a['DOI']),
            family=FAMILY_MAP[a['Class']],established_specimen_id='',
            quality_flag=QUALITY_FLAGS.get(i,('',''))[0],quality_evidence=QUALITY_FLAGS.get(i,('',''))[1]))
    # The curation did not establish any same-specimen IDs beyond original record regimes.
    # Unverified reuse suggestions are not promoted to verified identity.
    rows,edges=connected_components(rows)
    for r in rows:
        component=r['final_component']
        r['cohort_R']=True
        r['cohort_K']=component in ['bulk','total']
        r['cohort_B']=component=='bulk'
        r['cohort_T']=component=='total'
        r['cohort_KQ']=r['cohort_K'] and not r['quality_flag']
    return rows,edges


def prepare_cohorts(rows,edges):
    fields=['row_id','dataset_row_id','DOI','Formula','Ea','family','normalized_doi','normalized_formula',
        'normalization_status','connected_group_id','Type_of_conductivity','final_component',
        'classification_confidence','original_label_verification_status','quality_flag','quality_evidence',
        'cohort_R','cohort_K','cohort_B','cohort_T','cohort_KQ']
    write('cohort_manifest.csv',[{k:r[k] for k in fields} for r in rows],fields)
    write('quality_exclusion_manifest.csv',[dict(row_id=r['row_id'],flag=r['quality_flag'],
        evidence=r['quality_evidence'],in_primary_K=r['cohort_K'],excluded_from_KQ=r['cohort_K'])
        for r in rows if r['quality_flag']])
    write('connected_group_edges.csv',edges)
    families=sorted({r['family'] for r in rows})
    coverage=[]
    for cohort in ['K','B','T','R','KQ']:
        cr=[r for r in rows if r['cohort_'+cohort]]
        for family in families:
            fr=[r for r in cr if r['family']==family]
            counts=Counter(r['connected_group_id'] for r in fr)
            n=len(fr);g=len(counts)
            coverage.append(dict(cohort=cohort,family=family,rows=n,connected_groups=g,
                dois=len({r['normalized_doi'] for r in fr}),
                largest_group_rows=max(counts.values(),default=0),
                largest_group_fraction=max(counts.values(),default=0)/n if n else '',
                bulk_rows=sum(r['final_component']=='bulk' for r in fr),
                total_rows=sum(r['final_component']=='total' for r in fr),
                status='confirmatory' if g>=5 else 'secondary/exploratory' if g>=3 else 'case-study_only',
                P4_eligible=g>=5 and cohort!='R'))
    write('confirmatory_family_manifest.csv',coverage)
    return coverage


def grouped_splits(rr,protocol,n_splits,seed):
    from sklearn.model_selection import KFold,GroupKFold
    import numpy as np
    x=np.arange(len(rr))
    if protocol=='P1':
        yield from KFold(n_splits=n_splits,shuffle=True,random_state=seed).split(x)
    else:
        field='normalized_formula' if protocol=='P2' else 'connected_group_id'
        groups=[r[field] or 'UNRESOLVED:'+r['row_id'] for r in rr]
        yield from GroupKFold(n_splits=n_splits,shuffle=True,random_state=seed).split(x,groups=groups)


def build_splits(rows,coverage):
    splits=[]
    for cohort in ['K','B','T','R','KQ']:
        cr=[r for r in rows if r['cohort_'+cohort]]
        protocols=['P3'] if cohort=='KQ' else ['P1','P2','P3']
        for protocol in protocols:
            for repeat,seed in enumerate(SEEDS):
                for fold,(tr,te) in enumerate(grouped_splits(cr,protocol,5,seed)):
                    splits.append(dict(split_id=f'{cohort}_{protocol}_r{repeat}_f{fold}',cohort=cohort,
                        protocol=protocol,repeat=repeat,fold=fold,seed=seed,held_out_family='',
                        train_ids=[cr[i]['row_id'] for i in tr],test_ids=[cr[i]['row_id'] for i in te],purged_ids=[]))
        if cohort!='R':
            for item in coverage:
                if item['cohort']!=cohort or not item['P4_eligible']: continue
                family=item['family']
                touching={r['connected_group_id'] for r in rows if r['family']==family}
                train=[r for r in cr if r['connected_group_id'] not in touching]
                test=[r for r in cr if r['family']==family]
                purge=[r for r in cr if r['family']!=family and r['connected_group_id'] in touching]
                splits.append(dict(split_id=f'{cohort}_P4_{family}',cohort=cohort,protocol='P4',
                    repeat=0,fold=0,seed=MODEL_SEED,held_out_family=family,
                    train_ids=[r['row_id'] for r in train],test_ids=[r['row_id'] for r in test],
                    purged_ids=[r['row_id'] for r in purge]))
    byid={r['row_id']:r for r in rows}
    manifest=[]
    for s in splits:
        train=[byid[i] for i in s['train_ids']]
        inner=[]
        for fold,(tr,va) in enumerate(grouped_splits(train,s['protocol'],3,s['seed']+100+s['fold'])):
            inner.append(dict(inner_fold=fold,train_ids=[train[i]['row_id'] for i in tr],
                              validation_ids=[train[i]['row_id'] for i in va]))
        s['inner_splits']=inner
        for role in ['train','test','purged']:
            for rid in s[role+'_ids']:
                r=byid[rid]
                manifest.append(dict(split_id=s['split_id'],cohort=s['cohort'],protocol=s['protocol'],
                    repeat=s['repeat'],fold=s['fold'],held_out_family=s['held_out_family'],role=role,row_id=rid,
                    family=r['family'],normalized_doi=r['normalized_doi'],normalized_formula=r['normalized_formula'],
                    connected_group_id=r['connected_group_id']))
    write('split_manifest.csv',manifest)
    (OUT/'splits.json').write_text(json.dumps(splits,indent=2,ensure_ascii=False)+'\n')
    return splits
