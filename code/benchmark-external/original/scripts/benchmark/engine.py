"""Nested training and prediction, explicitly isolated from outer test targets."""
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from .design import ROOT,OUT,MODELS,GRIDS,FIXED,source_rows,sha
from .features import load


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values,dtype=np.float64).tobytes()).hexdigest()


class TrainingOnlyPreprocessor(BaseEstimator,TransformerMixin):
    def __init__(self,allowed_ids,scale=False):
        self.allowed_ids=allowed_ids;self.scale=scale

    def fit(self,X,y=None):
        ids=tuple(X.index)
        if set(ids)!=set(self.allowed_ids) or len(ids)!=len(self.allowed_ids):
            raise ValueError('Preprocessing fit input differs from authorized training partition')
        self.fit_ids_=ids
        self.imputer_=SimpleImputer(strategy='median',keep_empty_features=True)
        z=self.imputer_.fit_transform(X)
        self.filter_=VarianceThreshold(threshold=0.0)
        z=self.filter_.fit_transform(z)
        self.scaler_=StandardScaler() if self.scale else None
        if self.scaler_ is not None: self.scaler_.fit(z)
        self.n_output_features_=z.shape[1]
        return self

    def transform(self,X):
        z=self.filter_.transform(self.imputer_.transform(X))
        return self.scaler_.transform(z) if self.scaler_ is not None else z


def estimator(name,params):
    classes={'DummyRegressor':DummyRegressor,'Ridge':Ridge,'RBF-SVR':SVR,
             'ExtraTreesRegressor':ExtraTreesRegressor,'XGBoost':XGBRegressor}
    return classes[name](**FIXED[name],**params)


def fit_on_training(name,params,X,y,context,trace):
    if list(X.index)!=list(y.index): raise ValueError('Feature/target row misalignment')
    expected=context['expected_ids']
    if set(X.index)!=set(expected) or len(X)!=len(expected): raise ValueError('Unauthorized fit rows')
    if name=='DummyRegressor':
        model=estimator(name,params);model.fit(X,y)
        prepids=[];nfeat=0;imputer_sha='';scaler_n=0
    else:
        model=Pipeline([('preprocess',TrainingOnlyPreprocessor(tuple(expected),name in ['Ridge','RBF-SVR'])),
                        ('regressor',estimator(name,params))])
        model.fit(X,y)
        pp=model.named_steps['preprocess'];prepids=list(pp.fit_ids_)
        nfeat=pp.n_output_features_;imputer_sha=digest(pp.imputer_.statistics_)
        scaler_n=int(pp.scaler_.n_samples_seen_) if pp.scaler_ is not None else 0
    trace(dict(**{k:v for k,v in context.items() if k!='expected_ids'},model=name,params=params,
        fit_ids=list(X.index),preprocessing_fit_ids=prepids,n_features_after_filter=nfeat,
        target_fit_sha256=digest(y.to_numpy()),imputer_statistics_sha256=imputer_sha,
        scaler_training_rows=scaler_n,early_stopping=False))
    return model


def tune_and_fit(name,X_train,y_train,inner_splits,split_id,trace):
    """Outer test X/y are not accepted by this function. Candidate order breaks ties."""
    assert list(X_train.index)==list(y_train.index)
    allowed=set(X_train.index)
    for s in inner_splits:
        assert set(s['train_ids'])|set(s['validation_ids'])==allowed
        assert not set(s['train_ids'])&set(s['validation_ids'])
    candidate_results=[]
    if name=='DummyRegressor':
        best={};candidate_results=[dict(candidate=0,params='{}',inner_MAE_meV=None,selection='fixed training median')]
    else:
        for candidate,params in enumerate(GRIDS[name]):
            errors=[]
            for s in inner_splits:
                train=s['train_ids'];valid=s['validation_ids']
                context=dict(split_id=split_id,stage='inner_train',candidate=candidate,
                    inner_fold=s['inner_fold'],expected_ids=train)
                model=fit_on_training(name,params,X_train.loc[train],y_train.loc[train],context,trace)
                prediction=model.predict(X_train.loc[valid])
                errors.extend(np.abs(prediction-y_train.loc[valid].to_numpy()))
            candidate_results.append(dict(candidate=candidate,params=json.dumps(params,sort_keys=True),
                inner_MAE_meV=float(np.mean(errors)),selection='pooled inner-validation row MAE'))
        best_index=min(range(len(candidate_results)),key=lambda i:candidate_results[i]['inner_MAE_meV'])
        best=GRIDS[name][best_index]
    context=dict(split_id=split_id,stage='outer_refit',candidate=-1,inner_fold=-1,expected_ids=list(X_train.index))
    model=fit_on_training(name,best,X_train,y_train,context,trace)
    return model,best,candidate_results


def assert_frozen():
    path=OUT/'analysis_plan_v1.json'
    if not path.exists(): raise RuntimeError('No preregistration: fitting prohibited')
    expected=(OUT/'analysis_plan_v1.sha256').read_text().split()[0]
    if sha(path)!=expected: raise RuntimeError('Frozen plan modified')
    plan=json.loads(path.read_text())
    for filename,digest_ in plan['frozen_file_hashes'].items():
        if sha(ROOT/filename)!=digest_:
            raise RuntimeError(f'Frozen input/design changed: {filename}; versioned amendment required')
    return plan


def run():
    plan=assert_frozen()
    rows,_=source_rows();byid={r['row_id']:r for r in rows}
    X=load();y=pd.Series({r['row_id']:float(r['Ea']) for r in rows},name='primary_Ea_meV')
    splits=json.loads((OUT/'splits.json').read_text())
    checkpoints=OUT/'checkpoints';checkpoints.mkdir(exist_ok=True)
    tracepath=OUT/'fit_trace.jsonl'
    started=time.time()
    count=0
    with tracepath.open('a') as tracefile:
        def trace(e): tracefile.write(json.dumps(e,ensure_ascii=False)+'\n');tracefile.flush()
        for s in splits:
            for name in MODELS:
                count+=1
                path=checkpoints/(s['split_id']+'__'+name+'.json')
                if path.exists():
                    cached=json.loads(path.read_text())
                    assert cached['plan_sha256']==sha(OUT/'analysis_plan_v1.json')
                    continue
                train=s['train_ids'];test=s['test_ids']
                # No test y is passed to selection or fitting. Evaluation happens only in summarize.py.
                t=time.time()
                try:
                    model,best,candidates=tune_and_fit(name,X.loc[train],y.loc[train],s['inner_splits'],s['split_id'],trace)
                    predictions=model.predict(X.loc[test])
                    assert np.isfinite(predictions).all()
                    result=dict(plan_sha256=sha(OUT/'analysis_plan_v1.json'),split_id=s['split_id'],model=name,
                        selected_parameters=best,inner_search=candidates,test_ids=test,predictions=predictions.tolist(),
                        train_rows=len(train),train_groups=len({byid[i]['connected_group_id'] for i in train}),
                        purged_rows=len(s['purged_ids']),elapsed_seconds=time.time()-t,status='success')
                    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
                except Exception as exc:
                    (OUT/'run_failure.json').write_text(json.dumps(dict(split_id=s['split_id'],model=name,error=repr(exc)),indent=2))
                    raise
                print(f'{count}/{len(splits)*len(MODELS)} {s["split_id"]} {name}: predictions saved ({time.time()-t:.1f}s; total {(time.time()-started)/60:.1f} min)',flush=True)
    assert_frozen()
    print('All prespecified fits finished. Outer targets not scored by training engine.',flush=True)
