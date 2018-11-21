import numpy as np
import scipy as sp
from sys import stdout
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lgmres
import functools as fct
from scipy.sparse.linalg import ArpackError
from scipy.sparse.linalg import eigs
import warnings
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

#this is pretty bad! Fix it!
phiovereps=lambda dx,eta:  -1./2.*eta + 3./8. *dx*eta**2 - 5./16. *dx**2*eta**3 + 35./128. *dx**3*eta**4 - 63./256. *dx**4*eta**5 + 231./1024. *dx**5*eta**6 - 429./2048. *dx**6*eta**7 + \
            6435./32768. *dx**7*eta**8 - 12155./65536. *dx**8*eta**9 +46189./262144. *dx**9*eta**10 - 88179./524288. *dx**10*eta**11 + 676039./4194304. *dx**11*eta**12


def toMPS(Q,R,dx):
    """
    turn cMPS matrices Q,R into a mps matrix with 
    discretization parameter dx

    Parameters:
    ------------------
    Q    np.ndarray of shape (D,D)
         cmps matrix Q
    R    list of np.ndarray of shape (D,D)
         cmps matrices R
    dx:  float
         discretization parameter
    Returns:
    -----------------
    np.ndarray of shape (D,D,len(R))

    """
    
    if not (np.shape(Q)[0]==np.shape(Q)[1]):
        raise ValueError("Q matrix has to be square")
    if not (np.shape(R)[0]==np.shape(R)[1]):
        raise ValueError("R matrix has to be square")
    
    D=np.shape(Q)[0]
    matrix=np.zeros((D,D,len(R)+1)).astype(np.result_type(Q,*R))
    matrix[:,:,0]=np.eye(D)+dx*Q
    for n in range(len(R)):
        matrix[:,:,n+1]=np.sqrt(dx)*R[n]
    return matrix

def fromMPS(tensor,dx):
    """
    return the cMPS content of an mps tensor tensor
    Parameters:
    ------------
    tensor:  np.ndarray of shape (D,D,d)
             an mps tensor
    dx:      float
             discretization parameter
    Returns:
    -----------
    (Q,R)
    Q:   np.ndarray of shape (D,D)
         the cMPS Q matrix
    R:   list of length d of np.ndarray of shape (D,D)
         the cMPS R matrices
    """
    if tensor.shape[0]!=tensor.shape[1]:
        raise ValueError("mps tensor has to be square")
    
    D=tensor.shape[0]
    Q=np.copy((tensor[:,:,0]-np.eye(D))/dx)
    R=[np.copy(tensor[:,:,n]/np.sqrt(dx)) for n in range(1,tensor.shape[2])]
    return Q,R


def svd(mat,full_matrices=False,compute_uv=True,r_thresh=1E-14):
    """
    wrapper around numpy svd
    catches a weird LinAlgError exception sometimes thrown by lapack (cause not entirely clear, but seems 
    related to very small matrix entries)
    if LinAlgError is raised, precondition with a QR decomposition (fixes the problem)


    Parameters
    ----------
    mat:           array_like of shape (M,N)
                   A real or complex array with ``a.ndim = 2``.
    full_matrices: bool, optional
                   If True (default), `u` and `vh` have the shapes ``(M, M)`` and
                   (N, N)``, respectively.  Otherwise, the shapes are
                   (M, K)`` and ``(K, N)``, respectively, where
                   K = min(M, N)``.
    compute_uv :   bool, optional
                   Whether or not to compute `u` and `vh` in addition to `s`.  True
                   by default.

    Returns
    -------
    u : { (M, M), (M, K) } array
        Unitary array(s). The shape depends
        on the value of `full_matrices`. Only returned when
        `compute_uv` is True.
    s : (..., K) array
        Vector(s) with the singular values, within each vector sorted in
        descending order. The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.
    vh : { (..., N, N), (..., K, N) } array
        Unitary array(s). The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`. The size of the last two dimensions
        depends on the value of `full_matrices`. Only returned when
        `compute_uv` is True.
    """
    try: 
        [u,s,v]=np.linalg.svd(mat,full_matrices=False)
    except np.linalg.linalg.LinAlgError:
        [q,r]=np.linalg.qr(mat)
        r[np.abs(r)<r_thresh]=0.0
        u_,s,v=np.linalg.svd(r)
        u=q.dot(u_)
        print('caught a LinAlgError with dir>0')
    return u,s,v

def qr(mat,signfix):
    dtype=type(mat[0,0])
    q,r=np.linalg.qr(mat)
    if signfix=='q':
        sign=np.sign(np.diag(q))
        unit=np.diag(sign)
        return q.dot(unit),herm(unit).dot(r)
    if signfix=='r':
        sign=np.sign(np.diag(r))
        unit=np.diag(sign)
        return q.dot(herm(unit)),unit.dot(r)

def check_gauge(Q,R,gauge,thresh=1E-10,verbose=0):
    """
    checks if tensor obeys left or right orthogonalization;
    Parameters:
    ---------------
    Q:       np.ndarray of shape (D,D)
             an cMPS Q matrix
    R:       list of np.ndarray of shape (D,D)
             cMPS R matrices
    gauge:   int or str
             the gauge to be checked against: gauge can be in (1,'l','left') or (-1,'r','right')
             to check left or right orthogonality, respectively
    
    Returns: 
    ----------------
    a float giving the total deviation from orthonormality, i.e. ||(11|E-11|| or || E|11) -11||
    """
    
    Z=np.linalg.norm(transferOperator(Q,R,gauge,np.eye(Q.shape[0])))
    if (Z>thresh) and (verbose>0):
        print('check_gauge: cMPS is not {0} orthogonal with a residual of {1}'.format(gauge,Z) )
    return Z

def transferOperator(Q,R,direction,vector):
    """
    calculate the action of the cMPS transfer operator onto vector
    Parameters:
    ------------------
    Q:           np.ndarray of shape (D,D)
                 the Q-matrix of the cMPS
    R:           list of np.ndarray() of shape (D,D)
                 the R-matrices of the cMPS
    direction:   int
                 if direction in {1,'l','left'}: calculate the left-action
                 if direction in {-1,'r','right'}: calculate the right-action
    vector:      np.ndarray of shape (D*D) or (D,D)
                 the left or right vector; it has to be obtained from reshaping
                 a matrix of shape (D,D) into a vector.
                 Index convention of matrix: index "0" is on the unconjugated leg, index "1" 
                 on the conjugated leg

    Returns:
    -----------------------
    np.ndarray of shape vector.shape
    the result of applying the cMPS transfer operator to vector
    """
    
    D=np.shape(Q)[0]
    x=np.reshape(vector,(D,D))
    if direction in (1,'l','left'):
        out=np.transpose(Q).dot(x)+x.dot(np.conj(Q))
        for n in range(len(R)):
            out=out+np.transpose(R[n]).dot(x).dot(np.conj(R[n]))
        return np.reshape(out,vector.shape)

    elif direction in (-1,'r','right'):
        out=Q.dot(x)+x.dot(herm(Q))
        for n in range(len(R)):
            out=out+R[n].dot(x).dot(herm(R[n]))
        return np.reshape(out,vector.shape)

    
def pseudotransferOperator(Q,R,l,r,direction,vector):
    """
    computes the action of the pseudo transfer operator T^P=T-|r)(l|:
    
    direction in (1,'l','left'): (x| [ T-|r)(l| ]
    direction in (-1,'r','right'):   [ T-|r)(l| ] |x)
    
    for the (UNSHIFTED) transfer operator T, with (l| and |r) the left and right eigenvectors of T to eigenvalue 0

    Parameters: 
    ---------------------------
    Q:         np.ndarray of shape (D,D)
               the cMPS Q matrix
    R:         list() of np.ndarray() of shape (D,D)
               list() of cMPS R  matrices
    l:         np.ndarray of shape (D,D) or (D**2,)
               left  dominant eigenvector of cMPS transfer operator
    r:         np.ndarray of shape (D,D) or (D**2,)
               right dominant eigenvector of cMPS transfer operator
    direction: str() or int
               if direction in (1,'l','left'):   calculate the left action on vector
               if direction in (-1,'r','right'): calculate the right action on vector
    vector:    np.ndarray of shape (D**2) or (D,D)
               input vector
    Returns:
    ------------------------
    np.ndarray of shape vector.shape
    result of pseudotransferOperator as applied to vector
    """
    
    D=np.shape(Q)[1]
    x=np.reshape(vector,(D,D))
    
    if direction in (1,'l','left'):    
        return transferOperator(Q,R,direction,vector)-np.trace(np.transpose(x).dot(np.reshape(r,(D,D))))*np.reshape(l,vector.shape)
    elif direction in (-1,'r','right'):            
        return transferOperator(Q,R,direction,vector)-np.trace(np.transpose(np.reshape(l,(D,D))).dot(x))*np.reshape(r,vector.shape)


def inverseTransferOperator(Q,R,l,r,ih,direction,x0=None,tol=1e-10,maxiter=4000,**kwargs):
    """
    solves the equation systems

    [T-|r)(l|]|x)=|ih) for direction in (1,'l','left')

    or 

    (x|[T-|r)(l|]=(ih| for direction in (-1,'r','right')
    
    iteratively for |x) or (x|, using scipy's lgmres 
    

    Parameters: 
    ---------------------------
    Q:            np.ndarray of shape (D,D)
                  the cMPS Q matrix
    R:            list() of np.ndarray() of shape (D,D)
                  list of cMPS R  matrices
    l:            np.ndarray() of shape (D,D) or (D**2)
                  left  dominant eigenvector of cMPS transfer operator
    r:            np.ndarray() of shape (D,D) or (D**2)
                  right dominant eigenvector of cMPS transfer operator
    ih:           np.ndarray() of shape (D**2,) or (D,D)
                  right dominant eigenvector of cMPS transfer operator
    direction:    str() or int
                  if direction in (1,'l','left'):   calculate the left action on vector
                  if direction in (-1,'r','right'): calculate the right action on vector
    x0:           np.ndarray() of shape (D,D) or (D**2,), or None
                  in initial guess for the solution
    tol:          float (1E-12)
                  desired precision
    maxiter:      int (4000)
                  maximum iteration number 
    **kwargs:     additional parameters passed to lgmres (see documentation for detail

    Returns:
    ------------------------
    np.ndarray of shape vector.shape
    the solution |x) or (x| of the above equation
    """
    
    D=np.shape(Q)[1]
    mv=fct.partial(pseudotransferOperator,*[Q,R,l,r,direction])
    LOP=LinearOperator((D*D,D*D),matvec=mv,dtype=Q.dtype)
    [x,info]=lgmres(LOP,np.reshape(ih,D*D),x0=x0,tol=tol,maxiter=maxiter,**kwargs)
    while info<0:
        [x,info]=lgmres(LOP,np.reshape(ih,D*D),x0=np.random.rand(D*D).astype(Q.dtype),tol=tol,maxiter=maxiter,**kwargs)
    return np.reshape(x,ih.shape)


def TMeigs(Q,R,direction,numeig=6,init=None,maxiter=100000,tol=1e-12,ncv=40,which='LR',**kwargs):
    """
    calculate the dominant left or right eigenvector of the cMPS transfer operator
    Parameters:
    ------------------
    Q:           np.ndarray of shape (D,D)
                 the Q-matrix of the cMPS
    R:           list() of np.ndarray() of shape (D,D)
                 the R-matrices of the cMPS
    direction:   str or int
                 if direction in {1,'l','left'}: calculate the left steady state
                 if direction in {-1,'r','right'}: calculate the right steady state
    numeig:      int
                 number of eigenvalue-eigenvector pairs to be calculated
    init:        np.ndarray of shape (D,D) or (D**2) or None
                 initial guess for the eigenvector
    maxiter:     int
                 maximum number of iterations
    tol:         float
                 desired precision 
    ncv:         int
                 number of krylov vectors used in the solver
    which:       str
                 one of {'LM', 'SM' , 'LR' , 'SR' , 'LI' , 'SI'} see scipy documentation of eigs for details
    **kwargs:    additinal named parameters to be passed to eigs

    Returns:
    ------------------
    (eta,v)
    eta: np.result_type(Q,*R)
         eigenvalue
    v:   np.ndarray of dtype np.result_type(Q,*R)
         the dominant eigenvector
    """
    
    D=np.shape(Q)[0]
    dtype=np.result_type(Q,*R)
    mv=fct.partial(transferOperator,*[Q,R,direction])
    LOP=LinearOperator((D*D,D*D),matvec=mv,rmatvec=None,matmat=None,dtype=dtype)
    if init==None:
        v0=None
    else:
        v0=np.reshape(init,D*D)

    try:
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=v0,maxiter=maxiter,tol=tol,ncv=ncv,**kwargs)
        m=np.argmax(np.real(eta))
        while np.abs(np.imag(eta[m]))>1E-3:
            #numeig=numeig+1
            #print ('found TM eigenvalue with large imaginary part (ARPACK BUG); recalculating with larger numeig={0}'.format(numeig))
            print ('found TM eigenvalue eta ={0} with large imaginary part (ARPACK BUG); recalculating with a new initial state and LR'.format(eta))
            eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which='LR',v0=np.random.rand(D*D).astype(dtype),maxiter=maxiter,tol=tol,ncv=ncv,**kwargs)
            m=np.argmax(np.real(eta))
        return eta[m],np.reshape(vec[:,m],D*D)

    except ArpackError:
        print ('Arpack just threw an exception .... ' )
        return TMeigs(Q,R,dx,direction,numeig,np.random.rand(D*D).astype(dtype),maxiter,tol,ncv,which)

    

def regauge(Q,R,gauge='left',init=None,maxiter=100000,tol=1E-10,ncv=100,numeig=6,pinv=1E-200,thresh=1E-10,**kwargs):
    """
    regauge a cMPS into left or right orthogonal form

    Parameters:
    --------------------------
    Q:           np.ndarray of shape (D,D)
    R:           list of np.ndarrays of shape (D,D)
    gauge:       str
                 gauge in ['left','l','right','r']; output gauge of the cMPS
    init:        np.ndarray of shape (D,D) or (D**2,), or None
                 initial guess for the left or right dominant eigenvector
                 of the cMPS transfer operator
    maxiter:     int
                 maximum number of iterations
    tol:         float
                 desired precision of left and right dominant eigenvectors
                 of the cMPS transfer operators 
    ncv:         int
                 number of krylov vectors used in the solver
    numeig:      int
                 number of eigenvalue-eigenvector pairs to be calculated
    pinv:        float
                 pseudo-inverse cutoff; leave at default unless you know what you are
                 doing; setting it to too large values as compared to tol causes complete loss of orthogonality
    thresh:      float
                 if largest eigenvalue of the cMPS transferoperator has a imaginary part larger than thresh,
                 warning is raised
    **kwargs:    additinal named parameters to be passed to eigs

    Returns:
    ------------------------------
    gauge in ('l','left'): (r,y,Ql,Rl)
    gauge in ('r','right'): (l,x,Qr,Rr)

    r:     np.ndarray of shape (D,D)
           right dominant eigenvector of the cMPS transfer operator
    y:     np.ndarray of shape (D,D)
           gauge transformation matrix: Ql=y.dot(Q).dot(inv(y)), Rl[n]=y.dot(R[n]).dot(inv(y))
    Ql:    np.ndarray of shape (D,D)
           left orthogonal cMPS matrix
    Rl:    list of np.ndarray of shape (D,D)
           left orthogonal cMPS matrix

    l:     np.ndarray of shape (D,D)
           left dominant eigenvector of the cMPS transfer operator
    x:     np.ndarray of shape (D,D)
           gauge transformation matrix: Qr=inv(x).dot(Q).dot(x),Rr[n]=inv(x).dot(R[n]).dot(x)
    Qr:    np.ndarray of shape (D,D)
           right orthogonal cMPS matrix
    Rr:    list of np.ndarray of shape (D,D)
           right orthogonal cMPS matrix
    """
    dtype=np.result_type(Q,*R)
    if gauge in ('left','l'):
        [chi,chi2]=np.shape(Q)
        [eta,v]=TMeigs(Q,R,'left',numeig=numeig,init=init,maxiter=maxiter,tol=tol,ncv=ncv,which='LR',**kwargs)
        if np.abs(np.imag(eta))>thresh:
            warnings.warn('in regauge_with_trunc: found eigenvalue eta with large imaginary part: {0}'.format(eta),stacklevel=2)
        eta=np.real(eta)
        #normalization: this is a second order normalization
        Q-=eta/2.0*np.eye(chi)
        l=np.reshape(v,(chi,chi))
        #fix phase of l and restore the proper normalization of l
        l=l/np.trace(l)
        if dtype==float:
            l=np.real((l+herm(l))/2.0)
        if dtype==complex:
            l=(l+herm(l))/2.0

        eigvals,u=np.linalg.eigh(l)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals=eigvals/np.sum(eigvals)
        l=u.dot(np.diag(eigvals)).dot(herm(u))
        inveigvals=np.zeros(len(eigvals))
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0


        y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u)))
        invy=np.transpose(herm(u)).dot(np.diag(np.sqrt(inveigvals))).dot(np.transpose(u))

        Ql=y.dot(Q).dot(invy)
        Rl=[]
        for n in range(len(R)):
            Rl.append(y.dot(R[n]).dot(invy))
        return l,y,Ql,Rl

    if gauge in ('right','r'):
        [chi,chi2]=np.shape(Q)
        [eta,v]=TMeigs(Q,R,'right',numeig=numeig,init=init,maxiter=maxiter,tol=tol,ncv=ncv,which='LR',**kwargs)
        if np.abs(np.imag(eta))>thresh:
            warnings.warn('in regauge_with_trunc: found eigenvalue eta with large imaginary part: {0}'.format(eta),stacklevel=2)
        eta=np.real(eta)
        Q-=eta/2.0*np.eye(chi)
        r=np.reshape(v,(chi,chi))
        r=r/np.trace(r)
        if dtype==float:
            r=np.real((r+herm(r))/2.0)
        if dtype==complex:
            r=(r+herm(r))/2.0
        eigvals,u=np.linalg.eigh(r)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals/=np.sum(eigvals)
        l=u.dot(np.diag(eigvals)).dot(herm(u))

        inveigvals=np.zeros(len(eigvals))
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0

        r=u.dot(np.diag(eigvals)).dot(herm(u))
        x=u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u))
        invx=u.dot(np.diag(np.sqrt(inveigvals))).dot(herm(u))

        Rr=[]
        Qr=invx.dot(Q).dot(x)
        for n in range(len(R)):
            Rr.append(invx.dot(R[n]).dot(x))
        return r,x,Qr,Rr

    
def canonize(Q,R,linit=None,rinit=None,maxiter=100000,tol=1E-10,ncv=40,numeig=6,pinv=1E-200,trunc=1E-16,Dmax=100,thresh=1E-10,verbosity=0,**kwargs):
    """
    regauge a cMPS into left or right orthogonal form

    Parameters:
    --------------------------
    Q:           np.ndarray of shape (D,D)
    R:           list of np.ndarrays of shape (D,D)
    gauge:       str
                 gauge in ['left','l','right','r']; output gauge of the cMPS
    linit:       np.ndarray of shape (D,D) or (D**2,), or None
                 initial guess for the left dominant eigenvector
                 of the cMPS transfer operator
    rinit:       np.ndarray of shape (D,D) or (D**2,), or None
                 initial guess for the right dominant eigenvector
                 of the cMPS transfer operator
    maxiter:     int
                 maximum number of iterations
    tol:         float
                 desired precision of left and right dominant eigenvectors
                 of the cMPS transfer operators 
    ncv:         int
                 number of krylov vectors used in the solver
    numeig:      int
                 number of eigenvalue-eigenvector pairs to be calculated
    pinv:        float
                 pseudo-inverse cutoff; leave at default unless you know what you are
                 doing; setting it to too large values as compared to tol causes complete loss of orthogonality
    Dmax:        int
                 maximally allowed bond dimension; the cMPS is truncated down to this value
    trunc:       float
                 truncation threshold; if trunc>1E-15, Schmidt values smaller than trunc will be discarded
    Dmax:        int
                 if trunc>1E-15, the maximally allowed bond dimension is set to Dmax; 
                 the cMPS is truncated down to this value after Schmidtvalues < trunc have been discarded
    thresh:      float
                 if largest eigenvalue of the cMPS transferoperator has a imaginary part larger than thresh,
                 warning is raised
    **kwargs:    additinal named parameters to be passed to eigs

    Returns:
    ------------------------------
    (lam,Ql,Rl,Qr,Rr)

    lam:   np.ndarray of shape (D,D)
           right dominant eigenvector of the cMPS transfer operator
    Ql:    np.ndarray of shape (D,D)
           left orthogonal cMPS matrix
    Rl:    list of np.ndarray of shape (D,D)
           left orthogonal cMPS matrix
    Qr:    np.ndarray of shape (D,D)
           right orthogonal cMPS matrix
    Rr:    list of np.ndarray of shape (D,D)
           right orthogonal cMPS matrix
    rest:  list of float
           truncated Schmidt-values
    """
    dtype=np.result_type(Q,*R)    
    [chi ,chi2]=np.shape(Q)
    [etal,v]=TMeigs(Q,R,'left',numeig=numeig,init=linit,maxiter=maxiter,tol=tol,ncv=ncv,which='LR',**kwargs)
    if verbosity>0:
        print('left eigenvalue=',etal)

    if np.abs(np.imag(etal))>thresh:
        warnings.warn('in canonize: found eigenvalue eta with large imaginary part: {0}'.format(np.imag(etal)),stacklevel=2)
    etal=np.real(etal)

    l=np.reshape(v,(chi,chi))
    l=l/np.trace(l)
    if dtype==float:
        l=np.real((l+herm(l))/2.0)
    if dtype==complex:
        l=(l+herm(l))/2.0
    eigvals,u=np.linalg.eigh(l)
    eigvals[np.nonzero(eigvals<pinv)]=0.0
    eigvals=eigvals/np.sum(eigvals)
    l=u.dot(np.diag(eigvals)).dot(herm(u))

    inveigvals=np.zeros(len(eigvals))
    inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
    inveigvals[np.nonzero(eigvals<=pinv)]=0.0

    y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u)))
    invy=np.transpose(herm(u)).dot(np.diag(np.sqrt(inveigvals))).dot(np.transpose(u))

    Q-=etal/2.0*np.eye(chi)

    [etar,v]=TMeigs(Q,R,'right',numeig=numeig,init=rinit,maxiter=maxiter,tol=tol,ncv=ncv,which='LR',**kwargs)
    if verbosity>0:
        print('right eigenvalue=',etar)
    if np.abs(np.imag(etar))>thresh:
        warnings.warn('in canonize: found eigenvalue eta with large imaginary part: {0}'.format(np.imag(etar)),stacklevel=2)
    etar=np.real(etar)
    
    r=np.reshape(v,(chi,chi))
    r=r/np.trace(r)

    if dtype==float:
        r=np.real((r+herm(r))/2.0)
    if dtype==complex:
        r=(r+herm(r))/2.0

    eigvals,u=np.linalg.eigh(r)
    eigvals[np.nonzero(eigvals<pinv)]=0.0
    eigvals/=np.sum(eigvals)
    r=u.dot(np.diag(eigvals)).dot(herm(u))

    inveigvals=np.zeros(len(eigvals))
    inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
    inveigvals[np.nonzero(eigvals<=pinv)]=0.0


    r=u.dot(np.diag(eigvals)).dot(herm(u))
    x=u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u))
    invx=u.dot(np.diag(np.sqrt(inveigvals))).dot(herm(u))

    D=Q.shape[0]
    [U,lam,V]=svd(y.dot(x))        
    Z=np.linalg.norm(lam)
    lam=lam/Z        
    rest=[0.0]
    if trunc>1E-15:
        rest=lam[lam<=trunc]
        lam=lam[lam>trunc]
        rest=np.append(lam[min(len(lam),Dmax)::],rest)
        lam=lam[0:min(len(lam),Dmax)]
        U=U[:,0:len(lam)]
        V=V[0:len(lam),:]
        Z1=np.linalg.norm(lam)
        lam=lam/Z1
    Rl=[]
    Ql=Z*np.diag(lam).dot(V).dot(invx).dot(Q).dot(invy).dot(U)
    for n in range(len(R)):
        Rl.append(Z*np.diag(lam).dot(V).dot(invx).dot(R[n]).dot(invy).dot(U))
    
    Rr=[]
    Qr=Z*V.dot(invx).dot(Q).dot(invy).dot(U).dot(np.diag(lam))
    for n in range(len(R)):
        Rr.append(Z*V.dot(invx).dot(R[n]).dot(invy).dot(U).dot(np.diag(lam)))


    return lam,Ql,Rl,Qr,Rr,rest



def PiPiCorrMoronOrdered(Ql,Rl,r,dx,N,cutoff,psiinitial=None,psidaginitial=None):
    D=np.shape(Ql)[0]
    corr=np.zeros(N,dtype=type(Ql[0,0]))
    if np.any(psiinitial==None):
        vecpsi=-cutoff/2.0*(np.reshape(np.transpose(Rl),D*D))        
    else:
        vecpsi=psiinitial
    if np.any(psidaginitial==None):
        vecpsidag=-cutoff/2.0*(np.reshape(np.conj(Rl),D*D))
    else:
        vecpsidag=psidaginitial


    rdenspsi=np.tensordot(Rl,r,([1],[0]))
    rdenspsidag=np.tensordot(r,np.conj(Rl),([1],[1]))
    for n in range(N):
        if n%1000==0:
            stdout.write("\r %i/%i" %( n,N))
            stdout.flush()
        vecpsi=vecpsi+dx*transferOperator(Ql,[Rl],1,vecpsi)
        vecpsidag=vecpsidag+dx*transferOperator(Ql,[Rl],1,vecpsidag)
        corr[n]=np.tensordot(np.reshape(vecpsi,(D,D)),rdenspsi,([0,1],[0,1]))+\
                 np.tensordot(np.reshape(vecpsidag,(D,D)),rdenspsidag,([0,1],[0,1]))-\
                 2.0*np.tensordot(np.reshape(vecpsidag,(D,D)),rdenspsi,([0,1],[0,1]))
    return corr,vecpsi,vecpsidag

def calculateRelativisticCorrelators(Ql,Rl,r,cutoff,operators,dx,N,initial=None):
    """
    calculate correlators of (relativistic) field operators phi, pi, and partial phi for a 
    bosonic theory with a single species of bosons

    Parameters:
    --------------------
    Ql:        np.ndarray of shape (D,D)
               cMPS matrix in left orthogonal form
    Rl:        np.ndarray of shape (D,D)
               cMPS matrix in left orthogonal form
    r:         np.ndarray of shape (D,D)
               right reduced density matrix (i.e. right dominant eigenvector of the cMPS transfer operator)
    cutoff:    float
               enters definition of phi and pi via
               psi=sqrt(cutoff/2)phi +1/sqrt(2*cutoff)pi
    operators: list of length 2 of str
               each element in operators can be either of ['psi','psidag','n']
    dx:        float
               space increment, used to calculate the correlation at psi*(0)psi(n*dx)
    N:         int
               calculate correlator at points x=np.arange(N)*dx
    initial:   np.ndarray of shape (D**2,), or None
               you can feed the output second output of a prior call of the function back as initial state
               to resume calculation with different dx and N
    Returns:
    -----------------
    (corr,vec)
    corr: np.ndarray of shape (N,)
          the correlator
    vec: np.ndarray of shape (D**2,)
         result of the last evolution step
    """
    
    D=np.shape(Ql)[0]
    corr=np.zeros(N,dtype=type(Ql[0,0]))
    if operators[0]=='phi':
        if np.any(initial==None):
            vec=1.0/(2.0*cutoff)*(np.reshape(np.transpose(Rl)+np.conj(Rl),D*D))
        else:
            vec=initial
    elif operators[0]=='pi':
        if np.any(initial==None):
            vec=-cutoff/2.0*(np.reshape(np.transpose(Rl)-np.conj(Rl),D*D))
        else:
            vec=initial

    elif operators[0]=='p_phi':
        if np.any(initial==None):
            commQR=comm(Ql,Rl)
            vec=1/(2.0*cutoff)*(np.reshape(np.transpose(commQR)+np.conj(commQR),D*D))
        else:
            vec=initial
    else:
        raise ValueError("unknown operator {0}".format(operators[0]))
    
    if operators[1]=='phi':
        rdens=np.tensordot(Rl,r,([1],[0]))+np.tensordot(r,np.conj(Rl),([1],[1]))
    elif operators[1]=='pi':        
        rdens=np.tensordot(Rl,r,([1],[0]))-np.tensordot(r,np.conj(Rl),([1],[1]))
    elif operators[1]=='p_phi':
        commQR=comm(Ql,Rl)
        rdens=np.tensordot(commQR,r,([1],[0]))+np.tensordot(r,np.conj(commQR),([1],[1]))
    else:
        raise ValueError("unknown operator {0}".format(operators[1]))
        
    for n in range(N):
        if n%1000==0:
            stdout.write("\r %i/%i" %( n,N))
            stdout.flush()
        vec=vec+dx*transferOperator(Ql,[Rl],1,vec)
        corr[n]=np.tensordot(np.reshape(vec,(D,D)),rdens,([0,1],[0,1]))
    return corr,vec


def calculateCorrelators(Ql,Rl,r,operators,dx,N):
    """
    calculate correlators for a
    bosonic theory with a single species of bosons

    Parameters:
    --------------------
    Ql:        np.ndarray of shape (D,D)
               cMPS matrix in left orthogonal form
    Rl:        np.ndarray of shape (D,D)
               cMPS matrix in left orthogonal form
    r:         np.ndarray of shape (D,D)
               right reduced density matrix (i.e. right dominant eigenvector of the cMPS transfer operator)
    operators: list of length 2 of str
               each element in operators can be either of 'psi' or 'p','psidag' or 'pd','n'
    dx:        float
               space increment, used to calculate the correlation at psi*(0)psi(n*dx)
    N:         int
               calculate correlator at points x=np.arange(N)*dx

    Returns:
    -----------------
    np.ndarray of shape (N,)
    the correlator
    """
    
    D=np.shape(Ql)[0]
    corr=np.zeros(N,dtype=type(Ql[0,0]))
    if operators[0] in ('psidag','pd'):
        vec=np.reshape(np.conj(Rl),D*D)
    elif operators[0] in ('psidag_psidag','pd_pd'):
        vec=np.reshape(np.conj(Rl.dot(Rl)),D*D)
    elif operators[0] in ('psi','p'):        
        vec=np.reshape(np.transpose(Rl),D*D)
    elif operators[0] in ('psi_psi','p_p'):
        vec=np.reshape(np.transpose(Rl.dot(Rl)),D*D)
    elif operators[0] in ('n','pd_p','psidagger_psi'):
        vec=np.reshape(np.transpose(herm(Rl).dot(Rl)),D*D)
    elif operators[0] in ('p_pd','psi_psidagger'):
        raise ValueError('the requested operator psi_psidagger at position 0 has no finite expectation value')
        
    else:
        raise ValueError("unknown operator {0}".format(operators[0]))
    
    if operators[1] in ('psidag','pd'):
        rdens=np.tensordot(r,np.conj(Rl),([1],[1]))
    elif operators[1] in ('psidag_psidag','pd_pd'):
        rdens=np.tensordot(r,np.conj(Rl.dot(Rl)),([1],[1]))
    elif operators[1] in ('psi','p'):
        rdens=np.tensordot(Rl,r,([1],[0]))
    elif operators[1] in ('psi_psi','p_p'):
        rdens=np.tensordot(Rl.dot(Rl),r,([1],[0]))
    elif operators[1] in ('n','pd_p','psidagger_psi'):
        rdens=np.tensordot(np.tensordot(Rl,r,([1],[0])),np.conj(Rl),([1],[1]))
    elif operators[1] in ('p_pd','psi_psidagger'):
        raise ValueError('the requested operator psi_psidagger at position x has no finite expectation value')
        
    else:
        raise ValueError("unknown operator {0}".format(operators[1]))
        
    for n in range(N):
        if n%1000==0:
            stdout.write("\r %i/%i" %( n,N))
            stdout.flush()
        vec=vec+dx*transferOperator(Ql,[Rl],1,vec)
        corr[n]=np.tensordot(np.reshape(vec,(D,D)),rdens,([0,1],[0,1]))
    return corr


def calculateRenyiEntropy(Q,R,init,N,dx,alpha,eps=1E-8,Dmax=50,tol=1E-10):
    """
    calculate the Renyi entropies of  finite regions of length n*dx for n in range(N)
    """
    D=np.shape(Q)[0]
    #lam,Ql,Rl,Qr,Rr=regauge_old(Q,R,dx,gauge='symmetric',linitial=np.reshape(lamold,D*D),rinitial=np.reshape(np.eye(D),D*D),nmaxit=100000,tol=regaugetol)
    lam,Ql,Rl,Qr,Qr=canonize(Q,R,linit=init,rinit=init,maxiter=100000,tol=1E-10,ncv=40,numeig=6,pinv=1E-200,trunc=1E-16,Dmax=Q.shape[0],thresh=1E-10,verbosity=0,**kwargs)    

    etas=[]
    S=[]
    B=dcmps.toMPSmat(Qr,Rr,dx)
    ltensor=np.tensordot(np.diag(lam),B,([1],[0]))
    [D1r,D2r,dr]=np.shape(B)
    reachedmax=False
    for n in range(N):
        [D1l,D2l,dl]=np.shape(ltensor)
        mpsadd1=np.tensordot(ltensor,B,([1],[0])) #index ordering  0 T T 2
                                                  #                  1 3
        rho=np.reshape(np.tensordot(mpsadd1,np.conj(mpsadd1),([0,2],[0,2])),(dl*dr,dl*dr))   #  0  1
        eta,u=np.linalg.eigh(rho)
        inds=np.nonzero(eta>eps)
        indarray=np.array(inds[0])
        if len(indarray)<=Dmax:
            eta=eta[indarray]
        elif len(indarray)>Dmax:
            while len(indarray)>Dmax:
                indarray=np.copy(indarray[1::])
            eta=eta[indarray]
        etas.append(eta)
        R.append(1.0/(1.0-alpha)*np.log(np.sum(eta**alpha)))
        u_=u[:,indarray]
        utens=np.reshape(u_,(dl,dr,len(eta)))
        ltensor=np.tensordot(mpsadd1,np.conj(utens),([1,3],[0,1]))
    return etas,R
