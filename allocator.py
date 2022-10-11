import numpy as np
import scipy.linalg as la
from scipy.linalg import norm
import cvxpy as cp
def Find_Kappa(mu,m,s,sa,polytope_size=1000):# to solve the dual and find Kappa      (Eq.31)
    A=np.sin(np.outer(mu,np.linspace(0,np.pi,polytope_size)))
    l=cp.Variable(mu.shape[0])
    a=cp.Variable()
    objective = cp.Minimize((a**2+s**2/m*(cp.norm(cp.multiply(l,1/sa),2))**(2)-2*sum(cp.multiply(l,mu))))#
    constraints=[cp.norm(A.T@l,'inf')<=a]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=False)
    return -result*s**2/m,l.value
def find_global_extrema(l,mu,thres=1e-4): #find maximum of \rho to get the measurement positions  (Eq. 33)
    f=lambda x: sum(l[i]*np.sin(x*mu[i]) for i in range(len(l)))
    nl=l*mu
    n0=np.zeros(np.max(mu)+1)
    n0[mu]=nl
    p=np.poly1d(np.append(n0[::-1],n0[1:]))
    xr=p.roots
    x_cand=np.angle(np.array(xr)[np.abs(np.abs(xr)-1)<1e-2])
    x_cand=(x_cand[x_cand>=0])
    y=f(x_cand)
    ainf=la.norm(y,np.inf)
    inds=abs(abs(y)-ainf)<thres*ainf
    x,s=x_cand[inds],np.sign(y[inds])
    return ainf,np.sort(x),s[np.argsort(x)]

def coeff(xpos,mpos,mu,m,s,sa,wg=None):# Find weights with given measurement positions    (Eq- 29)  
    As=np.diag(sa)@np.sin(np.outer(mu,xpos))
    mus=np.diag(sa)@mu
    if wg is None:
        M=As.T@As+s**2*np.diag(1/mpos)
        wg=la.pinv(M)@As.T@mus
    r=la.norm(As@wg-mus)**2+s**2*la.norm(wg/np.sqrt(mpos))**2
    esys=(As@wg-mus)**2
    d2=np.sum(mu**2*sa**2)
    dg2=la.norm(As@wg)**2+s**2*la.norm(wg/np.sqrt(mpos))**2
    dgd2=(mu.T@np.diag(sa)@As@wg)**2
    return np.sqrt(r),wg,np.sqrt(esys),np.sqrt(dgd2/d2/dg2)
def discretize(m,mtot,ub=False,integer_meas=True):   #Round to integers. ub=True ensures that nothing is rounded to 0
    m*=mtot/la.norm(m,1)
    if integer_meas==False:
        return m
    mtot=int(mtot)
    mg=np.array(m,dtype=int)
    mr=m-mg
    Mc=sum(mg)
    r=round(sum(mr))
    if r==0: return mg
    mg[np.argsort(mr)[::-1][:r]]+=1
    if ub and mtot>=len(mg):
        for i in range(len(mg)):
            i=np.argmin(mg)
            if mg[i]==0:
                j=np.argmax(mg)
                mg[i]+=1
                mg[j]-=1
            else:
                break
    return mg

def solve_trig(mu,s2,sa2):# Solve the polynomial for SLGE. This function uses the variances
    cf=lambda x: sum(mu**2*sa2)-sum(np.sin(mu*x)*sa2*mu)**2/(sum(np.sin(mu*x)**2*sa2)+s2)
    nu=np.max(mu)
    M=np.zeros(nu*6+1,dtype=complex)
    os=3*nu
    for i in range(len(mu)):
        for j in range(len(mu)):
            c1=1/8 *sa2[i]*sa2[j]*mu[i]**2
            M[mu[i]+os]+=2*c1
            M[-(mu[i])+os]+=2*c1

            M[mu[i]+2*mu[j]+os]+=-c1
            M[-(mu[i]+2*mu[j])+os]+=-c1

            M[mu[i]-2*mu[j]+os]+=-c1
            M[-(mu[i]-2*mu[j])+os]+=-c1

            c2=-1/8*sa2[i]*sa2[j]*mu[j]*mu[i]
            M[(mu[i]-2*mu[j])+os]+=c2
            M[-(mu[i]-2*mu[j])+os]+=c2

            M[(mu[i]+2*mu[j])+os]-=c2
            M[-(mu[i]+2*mu[j])+os]-=c2
        c3=1/2*sa2[i]*mu[i]**2*s2
        M[(mu[i])+os]+=c3
        M[-(mu[i])+os]+=c3
    r=np.roots(M)
    x_cand=np.angle(r[np.absolute(np.absolute(r)-1)<1e-1])
    x_cand=np.sort(x_cand[x_cand>0])
    f_cand=[cf(xk) for xk in x_cand]
    ainf=min(f_cand)
    i=(np.arange(len(x_cand))[f_cand-ainf<1e-6])[0]
    return f_cand[i],x_cand[i]


#These function reduce the number of measuremetns setting to the minimum
def meas_bud(xpos,mu,m,s,sa,sgn):
    As=np.diag(sa)@np.sin(np.outer(mu,xpos))
    mus=np.diag(sa)@mu
    M=As.T@As+s**2/m*np.outer(sgn,sgn)
    wg=la.pinv(M)@As.T@mus
    r=la.norm(As@wg-mus)**2+s**2/m*la.norm(wg,1)**2
    return r,wg
def minimize_pos(ro,wpos,xpos,mu,m,s,sa,sgn,tol=1e-4):
    if len(xpos)>1:
        for i in np.arange(len(xpos)-1,0,-1):
            rn,w=meas_bud(np.delete(xpos,i),mu,m,s,sa,np.delete(sgn,i))
            if (rn-ro)/ro<=tol:
                return minimize_pos(rn,w,np.delete(xpos,i),mu,m,s,sa,np.delete(sgn,i),tol=tol)
    return ro,xpos,wpos

def minimize_posu(ro,xpos,mu,*r):
    if len(xpos)>1:
        for i in np.arange(len(xpos)-1,0,-1):
            S=np.sin(np.outer(mu,np.delete(xpos,i)))
            wn=la.pinv(S)@mu
            if la.norm(mu-S@wn)>1e-6:
                continue
            rn=la.norm(wn,1)
            if (rn-ro)/ro<=1e-8:
                return minimize_posu(rn,np.delete(xpos,i),mu)
    w=la.pinv(np.sin(np.outer(mu,xpos)))@mu
    return ro,xpos,w
def allocator(mu,m,sa=None,s=None,t=["BLGE","SLGE","ULGE"][0],full=False,integer_meas=True,ub=True,polytope_size=1000,thres=1e-4):
    '''
    Input:
        mu:           frequency differences (Positive integers)
        m:            number of total measurements (required to be even)
        s:            measurement noise std. deviation (!! not variance !!)
        sa:           expected std deviation (!! not variance !!) of the coefficient
        t:            type of method t  in ["BLGE","SLGE","ULGE"]
        full:         Returns additional information
        integer_meas: If the number of measurements is returned as integers
        ub:           ub=True means that the number of measurement positions will be at least 1. (Important for ULGE)
        polytope_size:Number of equations approximating the \infty norm in the convex program (Eq. 31)
        thres:        Relative error constant to remove sparsity. Only change if program does not converge)
        
    returns  
            xpos: measurement positions 
            mpos: measurements rounds for each site
            wpos: weight in the estimator
            if full
                cf :   Expected std. deviation error (root of mean squared error)
                esys:  Expected sytematic std. deviation error for each coefficient
                omega: Expected correlation Omega
    '''
    m//=2 
    mu,sa,nu=np.array(mu),np.array(sa),np.max(mu)
    
    if s is None: #without priors, default to ULGE
        t="ULGE"
    if t=="BLGE":
        r1,l=Find_Kappa(mu,m,s,sa,polytope_size=polytope_size)# find the ideal lambda
        af,xpos,sgn=find_global_extrema(l,mu)#find ideal measurement positions
        r2,w=meas_bud(xpos,mu,m,s,sa,sgn)#find ideal coefficients->measurement amounts
        r3,xpos,wpos=minimize_pos(r2,w,xpos,mu,m,s,sa,sgn,tol=thres)#removing unneccessary positions, which do not contribute
    elif t=="SLGE":
        r,x=solve_trig(mu,s**2/m,sa**2)
        xpos,mpos,wpos=np.array([x]),np.array([m]),np.array([1.])
    elif t=="ULGE":
        xp=np.pi/nu*(np.arange(nu)+1/2)
        r,xpos,wpos=minimize_posu(nu,xp,mu)
    mpos=discretize(np.abs(wpos),m,ub=ub,integer_meas=integer_meas)# round to nearest integer measurement
    xpos,mpos,wpos=xpos[mpos>0],mpos[mpos>0],wpos[mpos>0]#remove zeros from measurement positions
    xpos,mpos,wpos=np.append(-xpos[::-1],xpos),np.append(mpos[::-1],mpos),np.append(-wpos[::-1],wpos)/2# Adding the negative measuremetn psoitions
    if full:
        assert(s is not None)
        cf,wpos,esys,omega=coeff(xpos,mpos,mu,m,s,sa,wg=( wpos if (t=="ULGE") else None))
        return xpos,wpos,mpos,cf,esys,omega
    else:
        if t!="ULGE":
            wpos=coeff(xpos,mpos,mu,m,s,sa)[1]
        return xpos,wpos,mpos
