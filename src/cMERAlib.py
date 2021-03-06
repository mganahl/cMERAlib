#!/usr/bin/env python
import numpy as np
import sympy 
from sympy.core.symbol import Symbol
import functools as fct
import warnings
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lgmres
import scipy as sp

import src.cMERAcmpsfunctions as cmf
import math
import itertools
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))



def do_K_step(Q, R, lam, Gammas, Dmax=None, tol=1E-14, ncv=20, numeig=1, 
              warnthresh=1E-12, trunc=1E-14):
    
    Dmpo=Gammas[0][0].shape[0]
    Q_=np.kron(np.eye(Q.shape[0]),Gammas[0][0])+np.kron(Q,np.eye(Dmpo))+np.kron(R,Gammas[0][1])
    R_=np.kron(np.eye(R.shape[0]),Gammas[1][0])+np.kron(R,np.eye(Dmpo) + Gammas[1][1])
    if Dmax is None:
        # Dmpo=Gammas[0][0].shape[0]
        # Q_=np.kron(np.eye(Q.shape[0]),Gammas[0][0])+np.kron(Q,np.eye(Dmpo))+np.kron(R,Gammas[0][1])
        # R_=np.kron(np.eye(R.shape[0]),Gammas[1][0])+np.kron(R,np.eye(Dmpo) + Gammas[1][1])
        return Q_, R_, np.ones(Q_.shape[0]), 0.0
    else:
        lam, Ql, Rl, Qr, Rr, rest=cmf.canonize(Q_, [R_], linit=None, rinit=None,
                                               maxiter=100000, tol=tol,
                                               ncv=ncv, numeig=numeig, pinv=1E-200,
                                               thresh=warnthresh, trunc=trunc, Dmax=Dmax,
                                               verbosity=0)
        # lam, Ql, Rl, Qr, Rr, rest=cmf.canonize_cmps_cmpo(Q, [R], Gammas, linit=None, rinit=None,
        #                                                  maxiter=100000, tol=tol,
        #                                                  ncv=ncv, numeig=numeig, pinv=1E-200,
        #                                                  thresh=warnthresh, trunc=trunc, Dmax=Dmax,
        #                                                  verbosity=0)
        truncated_weight=np.sum(rest**2)
        return Ql, Rl[0], lam, truncated_weight
    
def toMPOmat(Gamma,dx,eta):
    D=np.shape(Gamma[0][0])[0]
    matrix=np.zeros((D,D,2,2)).astype(Gamma[0][0].dtype)
    matrix[:,:,0,0]=(np.eye(D)+dx*Gamma[0][0])
    matrix[:,:,0,1]=(np.sqrt(dx)*Gamma[0][1])
    matrix[:,:,1,0]=(np.sqrt(dx)*Gamma[1][0])    
    matrix[:,:,1,1]=Gamma[1][1]-dx*np.diag([0.0,-eta])
    return matrix

def checkWick(lam,Rl):
    Rc=Rl.dot(np.diag(lam))
    wick=0.25*(np.trace(herm(Rc).dot(herm(Rc)))*np.trace(Rc.dot(Rc))+np.trace(herm(Rc).dot(Rc))*np.trace(herm(Rc).dot(Rc)))
    full=np.trace(herm(Rc).dot(herm(Rc)).dot(Rc).dot(Rc))
    return wick,full


def checkWickTheorem(Ql,Rl,r,dx,N):
    dens=np.trace(Rl.dot(r).dot(herm(Rl)))
    cdag_c=cmf.calculateCorrelators(Ql,Rl,r,operators=['psidag','psi'],dx=dx,N=N)    
    c_cdag=np.conj(cdag_c)
    cdag_cdag=cmf.calculateCorrelators(Ql,Rl,r,operators=['psidag','psidag'],dx=dx,N=N)    
    c_c=np.conj(cdag_cdag)
    nn=cmf.calculateCorrelators(Ql,Rl,r,operators=['n','n'],dx=dx,N=N)        
    return nn-(cdag_cdag*c_c+dens**2+cdag_c*c_cdag)

def alpha(s,k):
    return np.sqrt((1+math.exp(2.0*s)*k**2)/(math.exp(2.0*s)*(1+k**2)))
    
def particle_density_exact(scale):
    return 1/(2*np.pi)*(np.cosh(scale)*sp.special.ellipk(1-np.exp(2*scale))-sp.special.ellipe(1-np.exp(-2*scale)))

def FreeBosonPiPiCorrelatorExact(scale,N=1000,dx=0.001):
    """
    calculate the Pi-Pi correlation function at scale "scale",
    for a state obtained from evolving an initial product state with
    UV cutoff 1.0 to a scale "scale".

    Parameters:
    --------------
    scale (float): the scale at which to calculate the correlator
    N (int > 0):   correlator is calculated at np.arange(-N, N) 8 dx
    dx (float):    spacing of points
    
    """
    #Discretize time t
    t=np.arange(-N,N,dx)
    #Define function
    f=1.0-alpha(scale,t)
    #Compute Fourier transform by numpy's FFT function
    g=np.fft.fft(f)
    #frequency normalization factor is 2*np.pi/dx
    w = np.fft.fftfreq(f.size)*2*np.pi/dx
    #In order to get a discretisation of the continuous Fourier transform
    #we need to multiply g by a phase factor
    g*=dx*np.exp(complex(0,1)*w*N)/(4*math.pi)
    pipiexact=g[w>=0]
    xexact=w[w>=0]
    return xexact,pipiexact

def FreeBosonPartialPhi_PartialPhiCorrelatorExact(steps,delta):
    #Discretize time t
    N=-1000.
    dx=0.001
    t=np.arange(N,-N,dx)
    #Define function
    s=np.abs(delta)*steps
    f=t**2/alpha(s,t)
    #Compute Fourier transform by numpy's FFT function
    g=np.fft.fft(f)
    #frequency normalization factor is 2*np.pi/dx
    w = np.fft.fftfreq(f.size)*2*np.pi/dx
    #In order to get a discretisation of the continuous Fourier transform
    #we need to multiply g by a phase factor
    g*=dx*np.exp(-complex(0,1)*w*N)/(2*math.pi)
    partialphipartialphiexact=g[w>=0]
    xexact=w[w>=0]
    return xexact,partialphipartialphiexact
    


#takes an mpo and a time step and creates a propagator mpo ala Zaletel
def createPropagatorMPO(mpo,delta):
    """
    takes an input operator H in mpo-form and creates the first order porpagator using
    the time step delta
    Parameters:
    ----------
    mpo:  np.ndarray of shape (M,M,d,d)
          input MPO representation of the operator H
    delta: float or complex
           time step

    Returns:
    ----------
    np.ndarray of shape (M-1,M-1,d,d)
    the propagator exp(delta H) in mpo form
    """
    M=mpo.shape[0]
    d=mpo.shape[2]
    #print(mpo.shape)
    #input()
    mat=np.transpose(mpo,(0,2,1,3))
    mat =np.reshape(mat,(d*M,d*M))
    C=mat[0:d,d:-d]
    D=mat[0:d,-d::]
    A=mat[d:-d,d:-d]
    B=mat[d:-d,-d::]

    prop=np.zeros((D.shape[0]+B.shape[0],D.shape[1]+C.shape[1])).astype(mat.dtype)
    prop[0:D.shape[0],0:D.shape[1]]=np.eye(D.shape[0])+delta*D
    prop[D.shape[0]::,0:D.shape[1]]=np.sqrt(delta)*B
    prop[0:D.shape[0],D.shape[1]::]=np.sqrt(delta)*C
    prop[D.shape[0]::,D.shape[1]::]=A
    #print(np.real(prop))

    out=np.transpose(np.reshape(prop,(M-1,d,M-1,d)),(0,2,1,3))
    return out

def getcMPO(mpo,dx):
    """
    takes a propagator mpo and a discretization parameter and returns a cmpo; note that even though 
    dx is finite, the result will be a true cmpo if the input mpo has been passed correctly
    note that Gamma[1][1] is set to be identically 0
    Parameters:
    ----------
    mpo:  np.ndarray of shape (M-1,M-1,d,d)
          input MPO representation of the propgator (e.g. exp(delta  H))
    dx:   float or complex
          discretization paramterer

    Returns:
    ----------
    Gamma: list of length 2 of list of length 2 of np.ndarray
    the cMPO representation of the propagator exp(delta H) in mpo form

    """
    M=mpo.shape[0]
    d=mpo.shape[2]    
    mat=np.reshape(np.transpose(mpo,(2,0,3,1)),(M*d,M*d))
    Gamma=[]
    for d1 in range(d):
        G=[]
        for d2 in range(d):
            if (d1==d2):
                if d1==0:
                    G.append((mat[d1*M:(d1+1)*M,d2*M:(d2+1)*M]-np.eye(M))/dx)
                else:
                    tmat=mat[d1*M:(d1+1)*M,d2*M:(d2+1)*M]
                    G.append(np.zeros(tmat.shape).astype(mpo.dtype))
            else:
                G.append(mat[d1*M:(d1+1)*M,d2*M:(d2+1)*M]/np.sqrt(dx))                
        Gamma.append(G)
    return Gamma

def freeBosoncMPO(Gammas, dx):
    """
    returns the discretized cMPO for the free Boson
    The index order convetions is:

                 3
                 |
                ___
               |   |
          0----|   |----1
               |___|
                 |
                 2
    An MPS is contracted from above:
                ___
             --|   |--
                ---
                 |
                ___
               |   |
           ----|   |----
               |___|
                 |

    """
    eye = np.eye(3)
    cmpo=[[eye + dx *Gammas[0][0],np.sqrt(dx) *Gammas[0][1]],
          [np.sqrt(dx) *Gammas[1][0],eye + dx *Gammas[1][1]]]
    return toMPOorder(cmpo)


def tocMPOorder(mpo):
    """
    takes an mpo and and permutes indices into cmpo order;
    Parameters:
    ----------
    mpo:  np.ndarray of shape (M,M,d,d)

    Returns:
    ----------
    Gamma: list of length d of list of length d of np.ndarray
    the MPO permuted into cMPO format

    """
    M=mpo.shape[0]
    d=mpo.shape[2]    
    #mat=np.reshape(np.transpose(mpo,(2,0,3,1)),(M*d,M*d))
    mat=np.reshape(np.transpose(mpo,(3,0,2,1)),(M*d,M*d))    
    Gamma=[]
    for d1 in range(d):
        G=[]
        for d2 in range(d):
            G.append(mat[d1*M:(d1+1)*M,d2*M:(d2+1)*M])
        Gamma.append(G)
    return Gamma
    
def toMPOorder(cMPO):
    """
    takes a list of lists of np.ndarray and 
    permutes it into MPO index order

    Parameters:
    -------------
    cMPO:    list of length d of list of length d of np.ndarray of shape (M,M)
             the cMPO matrices \Gamma_{ij}
    Returns:
    --------------
    np.ndarray of shape (M,M,d,d)
    the cMPO, permuted into MPO-index-order
    The index order convetions is:

                 3
                 |
                ___
               |   |
          0----|   |----1
               |___|
                 |
                 2
    An MPS is contracted from above:
                ___
             --|   |--
                ---
                 |
                ___
               |   |
           ----|   |----
               |___|
                 |

    
    """
    return np.transpose(np.array(cMPO),(2,3,1,0))


def density_density_entangler_propagator(α, β, Λ, δ):
    """
    β is the interaction strength: β \int n(x) n(y) exp(-Λ|x-y|) dx dy
    """
    ϵ = sympy.symbols('e')  
    b = sympy.Matrix(np.array([[0,1],[0,0]]))
    bd = b.T
    n = bd * b
    A = sympy.Identity(6)*(1 - ϵ*Λ)
    B = np.concatenate([np.sqrt(1j*α) * sympy.sqrt(ϵ) * bd, np.sqrt(-1j*α) * sympy.sqrt(ϵ) * b, np.sqrt(2 * β) * n],
                       axis=0)
    C = np.concatenate([np.sqrt(1j*α) * sympy.sqrt(ϵ) * bd, np.sqrt(-1j*α) * sympy.sqrt(ϵ) * b, np.sqrt(2 * β) * n],
                       axis=1)
    D = sympy.ZeroMatrix(2,2)
    tmp1 = np.concatenate([sympy.Identity(2) + δ * D, np.sqrt(δ) * C],axis=1)
    tmp2 = np.concatenate([np.sqrt(δ) * B, A],axis=1)
    tmp = np.concatenate([tmp1,tmp2], axis=0)
    cmpo = np.transpose(np.reshape(tmp,(4,2,4,2)),(0,2,1,3)) #(M, M, dout, din)
    Gammas=[[None,None],[None,None]]
    for d1, d2 in itertools.product([0,1],[0,1]):
        M = cmpo.shape[0]
        tmp = np.zeros((M, M), dtype=np.complex128)
        if d1!=d2:
            cmpo_tmp = cmpo[:,:,d1,d2]
            for a1, a2 in itertools.product(range(M),range(M)):
                tmp[a1,a2] = complex(cmpo_tmp[a1,a2].coeff('sqrt(e)'))
        elif (d1 == 0) and (d2 == 0):
            cmpo_tmp = cmpo[:,:,d1,d2] - sympy.Identity(M)
            for a1, a2 in itertools.product(range(M),range(M)):
                tmp[a1,a2] = complex(cmpo_tmp[a1,a2].coeff('e'))
                             
        elif (d1 == 1) and (d2 == 1):
            cmpo_tmp = cmpo[:,:,d1,d2] - sympy.Identity(M)
            for a1, a2 in itertools.product(range(M),range(M)):
                tmp[a1,a2] = complex(cmpo_tmp[a1,a2].subs([('e',0),('sqrt(e)',0)]))

        Gammas[d1][d2] = tmp
    return Gammas

def o1_o2_o3_o4_gammas(o1, o2, o3, o4, γ, Λ, δ):
    """
    γ is the interaction strength: γ \int_{w<x<y<z} ϕ(w)π(x)π(y)π(z) exp(-Λ|w-z|) dw dx dy dz
    """
    ϵ = sympy.symbols('e')  
    A = np.concatenate([np.concatenate([sympy.Identity(2)*(1 - ϵ*Λ), γ * sympy.sqrt(ϵ) *  sympy.Matrix(o2), sympy.ZeroMatrix(2,2)],axis=1),
                 np.concatenate([sympy.ZeroMatrix(2,2), sympy.Identity(2)*(1 - ϵ*Λ), γ *sympy.sqrt(ϵ) *  sympy.Matrix(o3)],axis=1),
                 np.concatenate([sympy.ZeroMatrix(2,2), sympy.ZeroMatrix(2,2), sympy.Identity(2)*(1 - ϵ*Λ)],axis=1)],
                 axis=0)
    B = np.concatenate([sympy.ZeroMatrix(2,2), sympy.ZeroMatrix(2,2), γ * sympy.sqrt(ϵ) * sympy.Matrix(o4)],
                       axis=0)
    C = np.concatenate([γ * sympy.sqrt(ϵ) * sympy.Matrix(o1),sympy.ZeroMatrix(2,2), sympy.ZeroMatrix(2,2)],
                       axis=1)
    D = sympy.ZeroMatrix(2,2)
    tmp1 = np.concatenate([sympy.Identity(2) + δ * D, np.sqrt(δ) * C],axis=1)
    tmp2 = np.concatenate([np.sqrt(δ) * B, A],axis=1)
    tmp = np.concatenate([tmp1,tmp2], axis=0)
    cmpo = np.transpose(np.reshape(tmp,(4,2,4,2)),(0,2,1,3)) #(M, M, dout, din)
    Gammas=[[None,None],[None,None]]
    for d1, d2 in itertools.product([0,1],[0,1]):
        M = cmpo.shape[0]
        tmp = np.zeros((M, M), dtype=np.complex128)
        if d1!=d2:
            cmpo_tmp = cmpo[:,:,d1,d2]
            for a1, a2 in itertools.product(range(M),range(M)):
                tmp[a1,a2] = complex(cmpo_tmp[a1,a2].coeff('sqrt(e)'))
        elif (d1 == 0) and (d2 == 0):
            cmpo_tmp = cmpo[:,:,d1,d2] - sympy.Identity(M)
            for a1, a2 in itertools.product(range(M),range(M)):
                tmp[a1,a2] = complex(cmpo_tmp[a1,a2].coeff('e'))
                             
        elif (d1 == 1) and (d2 == 1):
            cmpo_tmp = cmpo[:,:,d1,d2] - sympy.Identity(M)
            for a1, a2 in itertools.product(range(M),range(M)):
                tmp[a1,a2] = complex(cmpo_tmp[a1,a2].subs([('e',0),('sqrt(e)',0)]))

        Gammas[d1][d2] = tmp
    return Gammas


def free_boson_entangler_propagator(α, Λ, δ):
    """
    Generate the mpo representation of the entangling evolution for a free scalar, massless boson
    the implemented operator is  -1j*alpha/2\int dx dy exp(-Lambda*abs(x-y))*(psi(x)psi(y)-psidag(x)psidag(y))
    for alpha =Lambda/4, this gives the cMERA evolution operator for the free Boson theory with a UV-cutoff=Lambda; 
    The MPO is obtained from Gammas as 
    11 + dx * Gamma[0][0], sqrt(dx) Gamma[0][1]
    sqrt(dx) Gamma[1][0], 11 + Gamma[1][1]
    
    Parameters:
    -----------------
    alpha:    float or None
              the strength of the entangler; if None, alpha=cutoff/4
    Lambda:   float
              UV cutoff (see description above)
    delta:    float or complex
              time step of the entangling evolution; use np.real(delta) == 0, np.imag(delta) > 0 for lorentzian evolution

    Returns:
    -----------------
    Gamma: list of length 2 of list of length 2 of np.ndarray: 
           the cMPO matrices of the entangling propagator, in column major ordering
           The index convention is: 
           Gamma[a][b]:   a is the outgoing physical index, b is the incoming physical index
                          e.g. index a is contracted with the MPS:
                          Q_out = kron(11,Gamma[0][0])+kron(Q_in,11)+kron(R_in,Gamma[0][1])
                          R_out = kron(11,Gamma[1][0])+kron(R_in, 11 + Gamma[1][1])

    """
    ϵ = sympy.symbols('e')
    b = sympy.Matrix(np.array([[0,1],[0,0]]))
    bd = b.T
    n = bd * b
    A = sympy.Identity(4)*(1 - ϵ*Λ)
    B = np.concatenate([np.sqrt(1j*α) * sympy.sqrt(ϵ) * bd, np.sqrt(-1j*α) * sympy.sqrt(ϵ) * b],
                       axis=0)
    C = np.concatenate([np.sqrt(1j*α) * sympy.sqrt(ϵ) * bd, np.sqrt(-1j*α) * sympy.sqrt(ϵ) * b],
                       axis=1)
    D = sympy.ZeroMatrix(2,2)
    tmp1 = np.concatenate([sympy.Identity(2) + δ * D, np.sqrt(δ) * C],axis=1)
    tmp2 = np.concatenate([np.sqrt(δ) * B, A],axis=1)
    tmp = np.concatenate([tmp1,tmp2], axis=0)
    cmpo = np.transpose(np.reshape(tmp,(3,2,3,2)),(0,2,1,3)) #(M, M, dout, din)
    Gammas=[[None,None],[None,None]]
    for d1, d2 in itertools.product([0,1],[0,1]):
        M = cmpo.shape[0]
        tmp = np.zeros((M, M), dtype=np.complex128)
        if d1!=d2:
            cmpo_tmp = cmpo[:,:,d1,d2]
            for a1, a2 in itertools.product(range(M),range(M)):
                tmp[a1,a2] = complex(cmpo_tmp[a1,a2].coeff('sqrt(e)'))
        elif (d1 == 0) and (d2 == 0):
            cmpo_tmp = cmpo[:,:,d1,d2] - sympy.Identity(M)
            for a1, a2 in itertools.product(range(M),range(M)):
                tmp[a1,a2] = complex(cmpo_tmp[a1,a2].coeff('e'))
                             
        elif (d1 == 1) and (d2 == 1):
            cmpo_tmp = cmpo[:,:,d1,d2] - sympy.Identity(M)
            for a1, a2 in itertools.product(range(M),range(M)):
                tmp[a1,a2] = complex(cmpo_tmp[a1,a2].subs([('e',0),('sqrt(e)',0)]))

        Gammas[d1][d2] = tmp
    return Gammas

def freeEntanglingPropagator(cutoff,delta,alpha=None,dtype=complex,thresh=1E-14):
    """
    Generate the mpo representation of the entangling evolution for a free scalar, massless boson
    the implemented operator is  -1j*alpha/2\int dx dy exp(-cutoff*abs(x-y))*(psi(x)psi(y)-psidag(x)psidag(y))
    for alpha =cutoff/4, this gives the cMERA evolution operator for the free Boson theory with a UV-cutoff=cutoff; 
    the full prefactor is in this case -1j*cutoff/8

    Parameters:
    -----------------
    cutoff:   float
              UV cutoff (see description above)
    delta:    float or complex
              time step of the entangling evolution; use np.real(delta) == 0, np.imag(delta) > 0 for lorentzian evolution
    alpha:    float or None
              the strength of the entangler; if None, alpha=cutoff/4
    dtype:    type float or type complex or sympy.core.symbol.Symbol
              data type; us  for symbolic 
              evaluations, e.g. for using symbolic discretization dx later on

    Returns:
    -----------------
    Gamma: list of length 2 of list of length 2 of np.ndarray: 
           the cMPO matrices of the entangling propagator, in column major ordering
           The index convention is: 
           Gamma[a][b]:   a is the incoming physical index, b is the out-going physical index
                          e.g. index a is contracted with the MPS:
                          Q_out = kron(11,Gamma[0][0])+kron(Q_in,11)+kron(R_in,Gamma[1][0])
                          R_out = kron(11,Gamma[0][1])+kron(R_in,11)
    """
    if isinstance(dtype,tuple(sympy.core.all_classes)):
        #num=(delta*alpha)**(1/2)*sympy.exp(-1j*math.pi/4)
        #if sympy.Abs(sympy.re(delta))<thresh and sympy.im(delta)>0.0:
        #    num=(sympy.im(delta)*alpha)**(1/2)
        #elif sympy.Abs(sympy.re(delta))<thresh and sympy.im(delta)<0.0:
        #    num=(-sympy.im(delta)*alpha)**(1/2)*sympy.I
        #else:
        if alpha==None:
            alpha=cutoff/4.0
        
        num=(delta*alpha)**(1/2)*sympy.exp(-sympy.I*sympy.pi/4)
            
        G00=np.diag([sympy.S(0),-cutoff,-cutoff]).astype(dtype)*sympy.S(1.0)
        G11=np.diag([sympy.S(0),-cutoff,-cutoff]).astype(dtype)*sympy.S(1.0)
        
        G01=np.zeros((3,3)).astype(dtype)*sympy.S(1.0)
        G01[0,2]=num#np.sqrt(delta*alpha)*np.exp(-1j*math.pi/4)
        G01[2,0]=num#np.sqrt(delta*alpha)*np.exp(-1j*math.pi/4)
        
        G10=np.zeros((3,3)).astype(dtype)*sympy.S(1.0)
        G10[0,1]=num#np.sqrt(delta*alpha)*np.exp(-1j*math.pi/4)
        G10[1,0]=-num#np.sqrt(delta*alpha)*np.exp(-1j*math.pi/4)
    else:
        # if np.abs(np.imag(num))<thresh:
        #     num=np.real(num)
        # else:
        #     num=np.sqrt(delta*alpha)*np.exp(-1j*math.pi/4)
        if alpha==None:
            alpha=cutoff/4.0
        num=np.sqrt(delta*alpha)*np.exp(-1j*math.pi/4)

        G00=np.diag([0,-cutoff,-cutoff]).astype(dtype)
        G11=np.diag([0,-cutoff,-cutoff]).astype(dtype)
        
        G01=np.zeros((3,3)).astype(dtype)
        G01[0,2]=num#np.sqrt(delta*alpha)*np.exp(-1j*math.pi/4)
        G01[2,0]=num#np.sqrt(delta*alpha)*np.exp(-1j*math.pi/4)
        
        G10=np.zeros((3,3)).astype(dtype)
        G10[0,1]=num#np.sqrt(delta*alpha)*np.exp(-1j*math.pi/4)
        G10[1,0]=-num#np.sqrt(delta*alpha)*np.exp(-1j*math.pi/4)

    return [[np.copy(G00),np.copy(G01)],[np.copy(G10),np.copy(G11)]]

def density_density_interactingEntanglingPropagator(invrange,delta,inter,dtype=complex):
    G00=np.diag([0,-invrange]).astype(dtype)
    A11=np.eye(2).astype(dtype)
    A11[0,1]=np.sqrt(2*delta*inter)
    A11[1,0]=np.sqrt(2*delta*inter)
    return [[G00,np.zeros((2,2)).astype(dtype)],[np.zeros((2,2)).astype(dtype),A11]]

def interactingEntanglingPropagator(cutoff,invrange,delta,inter,operators=['phi','phi','phi','phi'],dtype=complex):
    """
    this is the implementation of 

    exp(delta K) with 
    K=inter*\int_{w,x,y,z} dw dx dy dz operators[0](w) operators[1](x) operators[2](y) operators[3](z) exp(-invrange(|w-x|+|x-y|+|y-z|))
    with phi=(b^{\dagger}+b)/sqrt(2*cutoff)

    Parameters:
    ---------------------------
    cutoff:    float 
               the UV cutoff of the state; this is what I call \nu in my notes
               psi(x)=sqrt(cutoff/2)phi(x)+i/sqrt(2*cutoff)pi(x)
    invrange:  float
               length scale of the exponential interaction
    delta:     complex, with imag(delta)>0
               time step
    inter:     float
               interaction strength
    operators: list of length 4 or 2 of str
               each element in operators can be either of ['phi','pi','psi','p','psidag','pd']
               where 'phi', 'pi', 'psi' or 'p' and 'psidag' or 'pd'  are the usual field operators of a bosonic field theory
              
    Returns:
    ---------------------------
    Gamma:a list of list containing the cMPO matrices of the propagator exp(delta K)
          note that Gamma[1][1] is 0!
    """
    #this  is a fake dx; drops out of equations later on
    dx=0.1
    if not all([o in ('phi','pi','psi','psidag','p','pd') for o in operators]):
        raise ValueError("unknown operators {}. each element in operators has to be one of ('phi','pi','psi','psidag')".format(np.array(operators)[[o not in ('phi','pi','psi','psidag','p','pd') for o in operators]]))
    
    c=np.zeros((2,2)).astype(dtype)
    c[0,1]=1.0    
    cdag=herm(c)

    ops={}
    ops['phi']=np.sqrt(dx)*(inter*24.0)**0.25*(c+herm(c))/np.sqrt(2*cutoff)
    ops['pi']=np.sqrt(dx)*(inter*24.0)**0.25*(c-herm(c))*np.sqrt(2*cutoff)/2.0j
    ops['psi']=np.sqrt(dx)*(inter*24.0)**0.25*c
    ops['p']=np.sqrt(dx)*(inter*24.0)**0.25*c    
    ops['psidag']=np.sqrt(dx)*(inter*24.0)**0.25*herm(c)
    ops['pd']=ops['psidag']
        
    interactingMPO=np.zeros((5,5,2,2)).astype(dtype)
    interactingMPO[0,0,:,:]=np.eye(2)

    interactingMPO[0,1,:,:]=ops[operators[0]]
    interactingMPO[1,1,:,:]=np.eye(2)*(1-dx*invrange)

    interactingMPO[1,2,:,:]=ops[operators[1]]
    interactingMPO[2,2,:,:]=np.eye(2)*(1-dx*invrange)

    interactingMPO[2,3,:,:]=ops[operators[2]]
    interactingMPO[3,3,:,:]=np.eye(2)*(1-dx*invrange)
    
    interactingMPO[3,4,:,:]=ops[operators[3]]
    interactingMPO[4,4,:,:]=np.eye(2)
    
    propaint=createPropagatorMPO(interactingMPO,delta)
    Gammasint=getcMPO(propaint,dx) 
    return Gammasint



def measure_energy_free_boson_with_cutoff(Ql,Rl,rdens,cutoff):
    operators=['dxpd_dxp','p_p','pd_pd','pd_p']
    prefactors=[1.0/cutoff,-cutoff/4.0,-cutoff/4.0,cutoff/2.0]
    ens=[pf*cmf.calculateLocalObservables(Ql,Rl,rdens,operator=op) for op,pf in zip(operators,prefactors)]
    #if not all(np.imag(ens)<1E-8):
    #    warnings.warn(f"imaginary energies found! {np.imag(ens)}",stacklevel=3)
    #    #raise TypeError("imaginary energies found!")
    return np.sum(np.real(ens))

def measure_energy_Lieb_Liniger(Ql,Rl,rdens,mass=0.5,mu=-0.5,g=1.0):
    operators=['dxpd_dxp','pd_pd_p_p','pd_p']
    prefactors=[1.0/(2*mass),g,mu]
    ens=[pf*cmf.calculateLocalObservables(Ql,Rl,rdens,operator=op) for op,pf in zip(operators,prefactors)]
    #if not all(np.imag(ens)<1E-8):
    #    warnings.warn(f"imaginary energies found! {np.imag(ens)}",stacklevel=3)        
    #    #raise TypeError("imaginary energies found!")
    return np.sum(np.real(ens))



