"""Original study functions, collected without their workspace-specific drivers."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from pymatgen.core import Composition
from reml import REML

# Source: A1_variance_decomposition/prepare.py

def compress(raw):
    a=raw.to_numpy(float); a[~np.isfinite(a)]=np.nan
    med=np.array([np.nanmedian(c) if np.isfinite(c).any() else 0 for c in a.T])
    a=np.where(np.isnan(a),med,a); mean=a.mean(0); std=a.std(0)
    keep=std > 0.001*np.maximum(1,np.abs(mean))
    z=(a[:,keep]-mean[keep])/std[keep]
    p=PCA(svd_solver='full').fit(z)
    need=int(np.searchsorted(np.cumsum(p.explained_variance_ratio_),.9)+1)
    n=min(need,20); scores=p.transform(z)[:,:n]/np.sqrt(p.explained_variance_[:n])
    return scores,dict(medians=med,means=mean,std=std,keep=keep,loadings=p.components_[:n],ratio=p.explained_variance_ratio_,need=need,n=n)

# Source: A1_variance_decomposition/run.py

def fit(d,model,component=True):
    names=['intercept'];xs=[np.ones(len(d))]
    if component:names+=['total_vs_bulk'];xs+=[(d.final_component=='total').to_numpy(float)]
    if model=='M4':
        pcs=[c for c in d if c.startswith('PC')];names+=pcs;xs+=[d[c].to_numpy() for c in pcs]
    labels={}
    if model in ['M1','M3','M4']:labels['source']=d.normalized_doi.to_numpy()
    if model in ['M2','M3','M4']:labels['family']=d.family.to_numpy()
    r=REML(d.Ea.to_numpy(),np.column_stack(xs),labels);out=r.fit();out['model']=model;out['fixed_names']=[names[j] for j in out['fixed_columns_kept']];out['rank_dropped_names']=[name for j,name in enumerate(names) if j not in out['fixed_columns_kept']];out['n_sources']=d.normalized_doi.nunique();out['n_families']=d.family.nunique();out['n_groups']=d.connected_group_id.nunique();return out

def draw_cluster(d,seed):
    rng=np.random.default_rng(seed);groups=np.array(sorted(d.connected_group_id.unique()));draw=rng.choice(groups,len(groups),replace=True);copies=[]
    for i,g in enumerate(draw):
        part=d[d.connected_group_id==g].copy();part['normalized_doi']=part.normalized_doi+'::copy'+str(i);part['bootstrap_copy']=i;copies.append(part)
    return pd.concat(copies,ignore_index=True),draw.tolist()

# Source: A2_applicability_domain/domain.py

def fit_geometry(train_X,test_X):
 """The complete fitted domain procedure. Signature accepts descriptors only."""
 train=np.asarray(train_X,float).copy();test=np.asarray(test_X,float).copy()
 train[~np.isfinite(train)]=np.nan;test[~np.isfinite(test)]=np.nan
 med=np.array([np.nanmedian(v) if np.isfinite(v).any() else 0. for v in train.T])
 train=np.where(np.isnan(train),med,train);test=np.where(np.isnan(test),med,test)
 mean=train.mean(0);sd=train.std(0,ddof=0);keep=sd>.001*np.maximum(1,np.abs(mean));assert keep.any(),'No usable training features'
 tr=(train[:,keep]-mean[keep])/sd[keep];te=(test[:,keep]-mean[keep])/sd[keep]
 pca=PCA(svd_solver='full',whiten=False).fit(tr);cumulative=np.cumsum(pca.explained_variance_ratio_);needed=int(np.searchsorted(cumulative,.90)+1);count=min(20,needed)
 trpc=pca.transform(tr)[:,:count];tepc=pca.transform(te)[:,:count]
 result={};artifacts={'imputation_median':med,'training_mean':mean,'training_std':sd,'retained_feature_mask':keep,'PCA_components':pca.components_[:count],'PCA_mean':pca.mean_,'explained_variance_ratio':pca.explained_variance_ratio_[:count],'PCs_required_for_90pct':np.array(needed),'train_scaled':tr,'train_PCs':trpc}
 for name,a,b in [('pca',trpc,tepc),('raw',tr,te)]:
  loo_matrix=cdist(a,a,metric='euclidean');np.fill_diagonal(loo_matrix,np.inf);loo=loo_matrix.min(1);loo_neighbor=loo_matrix.argmin(1)
  distances=cdist(b,a,metric='euclidean');nearest=distances.argmin(1);d=distances[np.arange(len(b)),nearest];q=float(np.quantile(loo,.95,method='linear'))
  r=d/q if q>0 else np.where(d==0,0.,np.inf)
  result[name]={'d':d,'q95':q,'r':r,'neighbor':nearest,'LOO':loo,'LOO_neighbor':loo_neighbor}
 return result,artifacts

def chemistry(formula):
 elements=sorted(e.symbol for e in Composition(formula).elements)
 return elements,'-'.join(e for e in elements if e!='O')

# Source: A2_1_domain_reference_robustness/geometry.py

def reference(coords,groups):
 """Pure geometry: only saved TRAIN coordinates and frozen TRAIN group identifiers."""
 x=np.asarray(coords,float);g=np.asarray(groups);assert len(np.unique(g))>=2
 distance=cdist(x,x,metric='euclidean');row=distance.copy();np.fill_diagonal(row,np.inf);group=distance.copy();group[g[:,None]==g[None,:]]=np.inf
 ri=row.argmin(1);gi=group.argmin(1);drow=row[np.arange(len(g)),ri];dgroup=group[np.arange(len(g)),gi]
 return {'row_neighbor':ri,'group_neighbor':gi,'drow':drow,'dgroup':dgroup,'qgroup':float(np.quantile(dgroup,.95,method='linear'))}

def normalized(d,q):return d/q if q>0 else np.where(d==0,0.,np.inf)

# Source: A2_1_domain_reference_robustness/analyze.py

def rho(x,y):
 if len(x)<3:return np.nan
 a=rankdata(x)-(len(x)+1)/2;b=rankdata(y)-(len(y)+1)/2;den=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/den) if den>0 else np.nan

def select(idx,d,c):return idx[np.lexsort((idx,d[idx]))[:int(np.ceil(len(idx)*c))]]
