#!/usr/bin/env python
import numpy as np
import functools as fct
import warnings
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lgmres
import src.cMERAcmpsfunctions as cmf
import math
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))


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
    

def FreeBosonPiPiCorrelatorExact(scale,N=-1000,dx=0.001):
    """
    calculate the Pi-Pi correlation function at scale "scale",
    for a state obtained from evolving an initial product state with
    UV cutoff 1.0 to a scale "scale".

    Parameters:
    --------------
    scale: float
           the scale at which to calculate the correlator
    t0:    int
    dt:    float
    
    """
    #Discretize time t
    t=np.arange(N,-N,dx)
    #Define function
    f=1.0-alpha(scale,t)
    #Compute Fourier transform by numpy's FFT function
    g=np.fft.fft(f)
    #frequency normalization factor is 2*np.pi/dx
    w = np.fft.fftfreq(f.size)*2*np.pi/dx
    #In order to get a discretisation of the continuous Fourier transform
    #we need to multiply g by a phase factor
    g*=dx*np.exp(-complex(0,1)*w*N)/(4*math.pi)
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

    
def freeEntanglingPropagator(cutoff,delta,alpha=None,dtype=complex):
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
              time step of the entangling evolution
    alpha:    float or None
              the strength of the entangler; if None, alpha=-1j*cutoff/8
    dtype:    type float or type complex
              data type 

    Returns:
    -----------------
    Gamma: list of length 2 of list of length 2 of np.ndarray: 
           the cMPO matrices of the entangling propagator
    """
    if alpha==None:
        alpha=cutoff/4.0
        
    G00=np.diag([0,-cutoff,-cutoff]).astype(dtype)
    G11=np.diag([0,-cutoff,-cutoff]).astype(dtype)

    G01=np.zeros((3,3)).astype(dtype)
    G01[0,2]=np.sqrt(delta*alpha)*np.exp(-1j*math.pi/4)
    G01[2,0]=np.sqrt(delta*alpha)*np.exp(-1j*math.pi/4)

    G10=np.zeros((3,3)).astype(dtype)
    G10[0,1]=np.sqrt(delta*alpha)*np.exp(-1j*math.pi/4)
    G10[1,0]=-np.sqrt(delta*alpha)*np.exp(-1j*math.pi/4)

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
    K=inter*\int_{w,x,y,z} dw dx dy dz operators[0] operators[1](x) operators[2](y) operators[3](z) exp(-invrange(|w-x|+|x-y|+|y-z|))
    with phi=(b^{\dagger}+b)/sqrt(2*cutoff)

    Parameters:
    cutoff: float 
            the UV cutoff of the state; this is what I call \nu in my notes
            \psi(x)=sqrt(cutoff/2)phi(x)+i/sqrt(2*cutoff)pi(x)
    invrange: float
            length scale of the exponential interaction
    delta:  complex, with imag(delta)>0
            time step
    inter: float
           interaction strength
    operators: list of length 4 or 2 of str
               each element in operators can be either of ['phi','pi','psi','psidag']
               where 'phi', 'pi', 'psi' and 'psidag' are the usual field operators of a bosonic field theory
              
    Returns:
    Gamma:a list of list containing the cMPO matrices of the propagator exp(delta K)
          note that Gamma[1][1] is 0!
    """
    dx=0.1
    if not all([o in ('phi','pi','psi','psidag') for o in operators]):
        raise ValueError("unknown operators {}. each element in operators has to be one of ('phi','pi','psi','psidag')".format(np.array(operators)[[o not in ('phi','pi','psi','psidag') for o in operators]]))
    
    c=np.zeros((2,2)).astype(dtype)
    c[0,1]=1.0    
    cdag=herm(c)
    

    ops={}
    ops['phi']=np.sqrt(dx)*(inter*24.0)**0.25*(c+herm(c))/np.sqrt(2*cutoff)
    ops['pi']=np.sqrt(dx)*(inter*24.0)**0.25*(c-herm(c))*np.sqrt(2*cutoff)/2.0j
    ops['psi']=np.sqrt(dx)*(inter*24.0)**0.25*c
    ops['psidag']=np.sqrt(dx)*(inter*24.0)**0.25*herm(c)


        
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








