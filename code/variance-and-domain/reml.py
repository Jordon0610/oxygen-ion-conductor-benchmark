"""Exact crossed-intercept Gaussian REML. No prior-phase model imports."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, qr
from scipy.optimize import minimize

def design_reduce(X):
    _,r,piv=qr(X,mode='economic',pivoting=True)
    d=np.abs(np.diag(r)); rank=int(np.sum(d > (d.max()*1e-10 if len(d) else 0)))
    keep=sorted(piv[:rank]); return X[:,keep],keep

def onehot(labels):
    u,inv=np.unique(labels,return_inverse=True)
    return np.eye(len(u))[inv]

class REML:
    def __init__(self,y,X,labels):
        self.y=np.asarray(y,float)/1000; self.X,self.keep=design_reduce(np.asarray(X,float))
        self.n,self.p=self.X.shape; self.df=self.n-self.p
        self.names=list(labels); self.Zs=[onehot(np.asarray(labels[key],str)) for key in self.names]
        self.qs=[z.shape[1] for z in self.Zs]; self.ends=np.cumsum([0]+self.qs)
        self.Z=np.concatenate(self.Zs,axis=1) if self.Zs else np.zeros((self.n,0))
        self.zz=self.Z.T@self.Z; B=np.column_stack([self.X,self.y]); self.bb=B.T@B; self.cb=self.Z.T@B
    def calc(self,lam,gradient=True,details=False):
        try:
            lam=np.asarray(lam,float); d=np.repeat(np.sqrt(lam),self.qs)
            if len(d):
                rr=np.eye(len(d))+self.zz*d[:,None]*d[None,:]; ch=cho_factor(rr,lower=True,check_finite=False)
                S=cho_solve(ch,np.diag(d),check_finite=False)*d[:,None]
                G=self.bb-self.cb.T@S@self.cb; ld=2*np.log(np.diag(ch[0])).sum()
            else:S=np.zeros_like(self.zz);G=self.bb;ld=0.
            A=G[:-1,:-1]; ach=cho_factor(A,lower=True,check_finite=False); Ai=cho_solve(ach,np.eye(self.p),check_finite=False)
            beta=Ai@G[:-1,-1]; Q=G[-1,-1]-G[-1,:-1]@beta
            if Q<=0 or self.df<=0:raise ValueError('nonpositive residual quadratic or degrees of freedom')
            scale=Q/self.df; nll=.5*(self.df*(np.log(2*np.pi)+1+np.log(scale))+ld+2*np.log(np.diag(ach[0])).sum())
            if gradient:
                U=self.cb[:,:-1]-self.zz@S@self.cb[:,:-1]
                T=self.zz-self.zz@S@self.zz-U@Ai@U.T
                v=self.cb[:,-1]-self.zz@S@self.cb[:,-1]-U@beta
                grad=np.array([.5*(np.trace(T[a:b,a:b])-np.sum(v[a:b]**2)/scale) for a,b in zip(self.ends[:-1],self.ends[1:])])
            else:grad=None
            if details:return nll,grad,scale,beta,Ai
            return nll,grad
        except (np.linalg.LinAlgError,ValueError,FloatingPointError):
            if details:raise
            return 1e100,np.zeros(len(lam))
    def hessian(self,lam):
        lam=np.asarray(lam); h=[]
        for i in range(len(lam)):
            step=1e-4*max(1,abs(lam[i])); lo=lam.copy();hi=lam.copy();lo[i]=max(0,lam[i]-step);hi[i]+=step
            h.append((self.calc(hi)[1]-self.calc(lo)[1])/(hi[i]-lo[i]))
        H=np.array(h).T;H=(H+H.T)/2
        ev=np.linalg.eigvalsh(H)
        singular=bool(ev.min()<=1e-8*max(1,ev.max()) or np.linalg.cond(H)>1e10)
        return singular,ev.tolist()
    def describe(self,lam,result):
        nll,grad,scale,beta,Ai=self.calc(lam,details=True)
        vv={'source_variance':0.,'family_variance':0.,'residual_variance':scale*1e6}
        for key,l in zip(self.names,lam): vv[key+'_variance']=float(l*scale*1e6)
        total=sum(vv.values());vpc={k.replace('variance','VPC'):v/total for k,v in vv.items()}
        boundary=any(vv[k+'_variance']<=1 or vpc[k+'_VPC']<=1e-6 for k in self.names)
        singular,ev=self.hessian(lam) if len(lam) else (False,[])
        pg=np.array(grad).copy(); pg[(np.array(lam)<=1e-8)&(pg>0)]=0
        norm=float(np.abs(pg).max()) if len(pg) else 0.
        vf=float(np.var(self.X@beta,ddof=1)*1e6)
        output={**vv,**vpc,'fixed_variance':vf,'marginal_R2':vf/(vf+total),'conditional_R2':(vf+vv['source_variance']+vv['family_variance'])/(vf+total),
            'n':self.n,'rank_X':self.p,'converged':bool(result.success and np.isfinite(nll) and nll<1e99),'boundary':boundary,'singular_hessian':singular,'upper_bound':bool(np.any(np.asarray(lam)>=999999)),
            'projected_gradient':norm,'stationarity_concern':norm>1e-3,'optimizer':result.method,'iterations':int(result.nit),'nll_internal':float(nll),'log_likelihood_meV':float(-nll-self.df*np.log(1000)),
            'hessian_eigenvalues':ev,'lambda':list(map(float,lam)),'fixed_coefficients':(beta*1000).tolist(),'fixed_SE':(np.sqrt(np.diag(Ai)*scale)*1000).tolist(),'fixed_columns_kept':self.keep}
        output['unstable']=bool(not output['converged'] or boundary or singular or output['upper_bound'] or output['stationarity_concern'])
        return output
    def fit(self):
        if not self.names:
            from types import SimpleNamespace
            r=SimpleNamespace(success=True,method='analytic OLS REML',nit=0)
            out=self.describe([],r);out['attempts']=[];return out
        attempts=[]; candidates=[]; start=np.ones(len(self.names));bounds=[(0,1e6)]*len(start)
        for method in ['L-BFGS-B','SLSQP','Powell']:
            if method=='L-BFGS-B':
                res=minimize(lambda t:self.calc(t),start,jac=True,bounds=bounds,method=method,options={'maxiter':1000,'ftol':1e-10,'gtol':1e-7})
            elif method=='SLSQP':
                res=minimize(lambda t:self.calc(t),start,jac=True,bounds=bounds,method=method,options={'maxiter':1000,'ftol':1e-10})
            else:
                res=minimize(lambda t:self.calc(t,gradient=False)[0],start,bounds=bounds,method=method,options={'maxiter':2000,'ftol':1e-10,'xtol':1e-8})
            res.method=method
            try:
                out=self.describe(res.x,res);candidates.append(out)
                attempts.append({k:out[k] for k in ['optimizer','iterations','converged','nll_internal','boundary','singular_hessian','projected_gradient','lambda'] }|{'message':str(res.message)})
                if not out['unstable']:break
                start=res.x.copy()
            except Exception as exc:attempts.append({'optimizer':method,'converged':False,'error':str(exc)})
        if not candidates:raise RuntimeError(str(attempts))
        valid=[c for c in candidates if c['converged']] or candidates
        out=min(valid,key=lambda r:r['nll_internal']);out['attempts']=attempts;return out


def self_test():
    from scipy.optimize._numdiff import approx_derivative
    rng=np.random.default_rng(42);n=45;X=np.c_[np.ones(n),rng.normal(size=(n,2))];y=rng.normal(700,300,n)
    lab={'source':np.arange(n)%9,'family':np.arange(n)%4};r=REML(y,X,lab);lam=np.array([.7,1.3])
    analytic=r.calc(lam)[1]; numeric=approx_derivative(lambda t:r.calc(t)[0],lam).ravel()
    assert np.allclose(analytic,numeric,atol=1e-6),(analytic,numeric)
    H=np.eye(n)+sum(l*z@z.T for l,z in zip(lam,r.Zs));Hi=np.linalg.inv(H);A=X.T@Hi@X;beta=np.linalg.solve(A,X.T@Hi@(y/1000));Q=(y/1000-X@beta)@Hi@(y/1000-X@beta)
    obj=.5*((n-3)*(np.log(2*np.pi)+1+np.log(Q/(n-3)))+np.linalg.slogdet(H)[1]+np.linalg.slogdet(A)[1])
    assert np.allclose(r.calc(lam)[0],obj,atol=1e-9)
    # Crossed covariance: same family different DOI and same DOI different family.
    assert H[0,4]==1.3 and H[0,9]==.7 and H[0,1]==0
    return {'analytic_gradient_vs_numerical':True,'lowrank_vs_dense_REML':True,'explicit_crossed_covariance':True}
if __name__=='__main__':print(self_test())
