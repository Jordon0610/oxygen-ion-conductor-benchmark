from pathlib import Path
import json,hashlib,datetime
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/phase3b'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a,dtype=np.float64).tobytes()).hexdigest()
def idhash(ids):return hashlib.sha256('\n'.join(ids).encode()).hexdigest()
def read(name):return json.loads((OUT/name).read_text())
def js(path,d):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)
def plan():
 p=read('analysis_plan_phase3b.json');assert sha(OUT/'analysis_plan_phase3b.json')==read('preregistration_status.json')['plan_sha256'];return p
class Prep:
 def __init__(self,X):
  self.ids=list(X.index);self.input_columns=list(X.columns);self.imputer=SimpleImputer(strategy='median',keep_empty_features=True)
  z=self.imputer.fit_transform(X);self.filter=VarianceThreshold(0);self.filter.fit(z)
 def transform(self,X):
  assert list(X.columns)==self.input_columns
  return self.filter.transform(self.imputer.transform(X))
 def trace(self):return dict(preprocessing_fit_id_hash=idhash(self.ids),median_hash=digest(self.imputer.statistics_),retained_mask=self.filter.get_support().tolist(),input_dimension=len(self.input_columns),output_dimension=int(self.filter.get_support().sum()))
class DFTPrep:
 """Exact frozen Phase3A.1 tree preprocessing, including floating-point std recipe."""
 def __init__(self,X):
  self.ids=list(X.index);self.input_columns=list(X.columns)
  a=X.to_numpy(float);self.median=np.array([np.nanmedian(a[:,j]) if np.isfinite(a[:,j]).any() else 0. for j in range(a.shape[1])])
  a=np.where(np.isfinite(a),a,self.median);self.mask=a.std(axis=0)>0
  assert self.mask.any()
 def transform(self,X):
  assert list(X.columns)==self.input_columns
  a=X.to_numpy(float);return np.where(np.isfinite(a),a,self.median)[:,self.mask]
 def trace(self):return dict(preprocessing_fit_id_hash=idhash(self.ids),median_hash=digest(self.median),retained_mask=self.mask.tolist(),input_dimension=len(self.input_columns),output_dimension=int(self.mask.sum()),implementation='frozen_Phase3A1_std_gt_zero')
def estimator(name,params,dft=False):
 seed=161803 if dft else 31415
 if name=='ExtraTreesRegressor':return ExtraTreesRegressor(n_estimators=128,max_depth=None,bootstrap=False,n_jobs=2,random_state=seed,**params)
 if name=='XGBoost':return XGBRegressor(n_estimators=150,objective='reg:squarederror',tree_method='hist',subsample=1,colsample_bytree=1,reg_lambda=10,n_jobs=2,random_state=seed,verbosity=0,**params)
 raise ValueError(name)
def fit_predict(name,params,X,y,V,scope,dft=False):
 assert list(X.index)==list(y.index);assert y.name in (['Ef_min_sourcefiltered_q0','Ef_mean_sourcefiltered_q0'] if dft else ['Ea_meV'])
 if dft:assert X.shape[1]==139 and not any(c.startswith('component_') for c in X.columns)
 prep=DFTPrep(X) if dft else Prep(X);model=estimator(name,params,dft);model.fit(prep.transform(X),y.to_numpy());pred=model.predict(prep.transform(V));assert np.isfinite(pred).all()
 return pred,dict(scope=scope,model=name,params=params,train_id_hash=idhash(list(X.index)),training_target_hash=digest(y),validation_id_hash=idhash(list(V.index)),target_name=y.name,**prep.trace())
