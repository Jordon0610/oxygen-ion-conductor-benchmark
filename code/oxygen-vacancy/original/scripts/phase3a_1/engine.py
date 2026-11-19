"""Phase 3A.1 DFT-only nested fitting; no experimental outcomes accepted."""
import json,hashlib,datetime
import numpy as np,pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from prepare import ROOT,OUT,sha,js,TARGETS
ALLOWED_TARGETS=set(TARGETS)
EVENTS=[]
class FoldPreprocessor:
 def __init__(self,scaled):self.scaled=scaled
 def fit(self,X):
  self.fit_ids=tuple(X.index)
  a=X.to_numpy(float);self.median=np.array([np.nanmedian(a[:,j]) if np.isfinite(a[:,j]).any() else 0. for j in range(a.shape[1])])
  a=np.where(np.isfinite(a),a,self.median);self.mean=a.mean(axis=0);self.std=a.std(axis=0)
  self.threshold=1e-3*np.maximum(1.,np.abs(self.mean)) if self.scaled else np.zeros(a.shape[1])
  self.mask=self.std>self.threshold
  if not self.mask.any():raise ValueError('No informative training features')
  return self
 def transform(self,X):
  a=X.to_numpy(float);a=np.where(np.isfinite(a),a,self.median)
  return ((a-self.mean)/np.where(self.std>0,self.std,1))[:,self.mask] if self.scaled else a[:,self.mask]

def make_model(name,params,seed):
 if name=='DummyRegressor':return DummyRegressor(strategy='median')
 if name=='Ridge':return Ridge(**params)
 if name=='RBF-SVR':return SVR(kernel='rbf',**params)
 if name=='ExtraTreesRegressor':return ExtraTreesRegressor(n_estimators=128,n_jobs=2,random_state=seed,**params)
 if name=='XGBoost':return XGBRegressor(n_estimators=150,n_jobs=2,random_state=seed,objective='reg:squarederror',tree_method='hist',subsample=1,colsample_bytree=1,reg_lambda=10,**params)
 raise ValueError(name)

def fit_predict_dft(Xtrain,ytrain,Xvalid,name,params,seed,scope,allowed_train_ids):
 assert ytrain.name in ALLOWED_TARGETS,'Only DFT host targets may enter training'
 assert list(Xtrain.index)==list(ytrain.index)
 assert set(Xtrain.index)==set(allowed_train_ids)
 assert set(Xtrain.index).isdisjoint(Xvalid.index)
 assert not any(c in Xtrain.columns for c in ['Ea','EaHT','family','DOI','final_component'])
 prep=FoldPreprocessor(name in ['Ridge','RBF-SVR']).fit(Xtrain)
 assert set(prep.fit_ids)==set(allowed_train_ids)
 model=make_model(name,params,seed);model.fit(prep.transform(Xtrain),ytrain.to_numpy())
 pred=model.predict(prep.transform(Xvalid))
 assert np.isfinite(pred).all(),'Nonfinite prediction; stop rather than clip'
 EVENTS.append(dict(scope=scope,model=name,params=params,target=ytrain.name,train_ids=list(Xtrain.index),
   validation_ids=list(Xvalid.index),preprocessing_fit_ids=list(prep.fit_ids),near_constant_threshold='std <= 1e-3 * max(1,abs(mean))' if prep.scaled else 'exact constant only',
   kept_features=int(prep.mask.sum()),training_target_sha256=hashlib.sha256(ytrain.to_numpy().tobytes()).hexdigest()))
 return pred

def select_hyperparameters(Xtrain,ytrain,inner_splits,name,candidates,seed,scope):
 # This function receives outer-training data only. It cannot inspect outer test targets.
 scores=[]
 for ci,params in enumerate(candidates):
  losses=[]
  for j,(it,iv) in enumerate(inner_splits):
   xp=Xtrain.loc[it];yp=ytrain.loc[it]
   pred=fit_predict_dft(xp,yp,Xtrain.loc[iv],name,params,seed,f'{scope}/inner{j}/candidate{ci}',it)
   losses.extend(np.abs(ytrain.loc[iv].to_numpy()-pred))
  scores.append(float(np.mean(losses)))
 best=int(np.argmin(scores));return candidates[best],scores

def run():
 planpath=OUT/'analysis_plan_phase3a_1.json';plan=json.loads(planpath.read_text());planhash=sha(planpath)
 assert not (OUT/'predictions.csv').exists() and not (OUT/'fit_started.json').exists(),'No automatic repeat fitting'
 for path,digest in plan['frozen_input_hashes'].items():assert sha(ROOT/path)==digest,path
 X=pd.read_csv(OUT/'composition_features.csv',index_col='formula_id',float_precision='round_trip')
 H=pd.read_csv(OUT/'source_clean_host_targets.csv',float_precision='round_trip')
 splits=json.loads((OUT/'splits.json').read_text());rows=[];selections=[]
 js(OUT/'fit_started.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=planhash))
 # All primary B results finish before any sensitivity fit.
 for population in ['B','C1','C2','C3']:
  h=H[H.population.eq(population)].set_index('formula_id')
  for target in TARGETS:
   y=h[target].copy();y.name=target
   for protocol in (['S1','S2'] if population=='B' else ['S2']):
    for fold in splits[population][protocol]:
     tr=fold['train'];te=fold['test'];inner=[(x['train'],x['test']) for x in fold['inner']]
     for name,candidates in plan['models'].items():
      scope=f'{population}/{target}/{protocol}/fold{fold["fold"]}'
      if name=='DummyRegressor':params={};scores=[]
      else:params,scores=select_hyperparameters(X.loc[tr],y.loc[tr],inner,name,candidates,plan['model_seed'],scope)
      pred=fit_predict_dft(X.loc[tr],y.loc[tr],X.loc[te],name,params,plan['model_seed'],scope+'/outer_refit',tr)
      for formula_id,value in zip(te,pred):rows.append(dict(population=population,formula_id=formula_id,chemical_system=h.loc[formula_id,'chemical_system'],target=target,protocol=protocol,fold=fold['fold'],model=name,
       observed_eV=float(y.loc[formula_id]),predicted_eV=float(value),absolute_error_eV=float(abs(y.loc[formula_id]-value)),squared_error_eV=float((y.loc[formula_id]-value)**2)))
      selections.append(dict(population=population,target=target,protocol=protocol,fold=fold['fold'],model=name,params=params,inner_candidate_MAE=scores))
      print(scope,name,'completed',flush=True)
  pd.DataFrame(rows).to_csv(OUT/'predictions_in_progress.csv',index=False,float_format='%.17g')
  js(OUT/(population+'_completed.json'),dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 pd.DataFrame(rows).to_csv(OUT/'predictions.csv',index=False,float_format='%.17g');(OUT/'predictions_in_progress.csv').unlink(missing_ok=True)
 js(OUT/'fit_trace.json',EVENTS);js(OUT/'hyperparameter_selections.json',selections)
 assert sha(planpath)==planhash
 js(OUT/'fit_completed.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=planhash,fit_events=len(EVENTS),prediction_sha256=sha(OUT/'predictions.csv')))
if __name__=='__main__':run()
