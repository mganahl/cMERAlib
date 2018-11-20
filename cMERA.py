#!/usr/bin/env python
import numpy as np
import pickle
import os,sys,select
import datetime
import src.cMERAcmpsfunctions as cmf
import src.cMERAlib as cmeralib
import matplotlib.pyplot as plt
import math
import argparse
import warnings
import re
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

class cMERA(object):
    """
    a class for simulating a cMERA evolution
    """
    def __init__(self,cutoff=1.0,alpha=None,inter=0.0,invrange=1.0,operators=['n','n'],delta=1E-3j,nwarmup=2,Dmax=16,dtype=complex):
        """
        initialize a cMERA evolution for a scalar boson
        Parameters:
        --------------------
        cutoff:    float (1.0)
                   UV cutoff of the cMERA
                   cutoff enters in the definition of the free entangler K0 (see below) 
                   as well as in the definition of the interacting entangler via the definition of 
                   psi(x) =\sqrt(cutoff/2)phi(x)+1/sqrt(2*cutoff)pi(x)
        alpha:     float or None
                   prefactor of the free entangler, if None, alpha=cutoff/4 is used
                   the free entangler has the form 
                   K0=alpha\int dx dy exp(-cutoff*abs(x-y)) :pi(x)phi(y):
                     =-1j*alpha/2\int dx dy exp(-cutoff*abs(x-y))*(psi(x)psi(y)-psidag(x)psidag(y))
                   to obtain the correct entangler for the ground-state of free boson with cutoff, use alpha=cutoff/4 (default)
        inter:     float
                   the interaction strength of the entangler
        invrange:  float
                   inverse length scale of the interaction
        operators: list of str
                   operators used to construct entangling propagator
                   case 1: for operators = ['n','n'], the entangler is given by inter * \int dx dy n(x) n(y) exp(-invrange*abs(x-y))
                           if len(operators)==2, operators=['n','n'] is the only allowed choice
                   case 2: for operators = [o1,o2,o3,o4], the entangler is given by inter * \int dw dx dy dx o1(w) o2(x) o3(y) o4(z) exp(-invrange abs(w-z))
                           if len(operators)==4, each element in operators can be one of the following :'phi','pi','psi','psidag'
        delta:     complex 
                   the step-size of the evolution
        nwarmup:   int
                   number of warmups steps
        Dmax:      int
                   maximum bond dimension
        dtype:     type float or type complex (complex)
                   the data type of the cMERA/cMPS matrices

        """
        
        self.dtype=dtype
        self.Dmax=Dmax
        self.scale=0.0
        self.iteration=0
        self.cutoff=cutoff
        self.Ql=np.ones((1,1)).astype(self.dtype)
        self.Rl=np.zeros((1,1)).astype(self.dtype)
        self.lam=np.array([1.0])
        self.D=len(self.lam)
        self.truncated_weight=0.0
        self.Gamma=cmeralib.freeEntanglingPropagator(cutoff=cutoff,delta=delta,alpha=alpha)

        for n in range(nwarmup):
            self.doStep(cutoff=cutoff,alpha=alpha,inter=inter,invrange=invrange,operators=operators,delta=delta,truncAfterFree=False,truncAfterInt=False)
            #do a warmup evolution without any truncation
            #problems can arise if truncation threshold of the evolution is so large that it truncates away any new schmidt values; in this case
            #the state will stay a product state for all times. A way to circumvent this problem is to either choose a smaller truncation threshold
            #or do a warmup run without truncation. This way, the evolution can introduce Schmidt-values above the truncation threshold
        #self.lam,self.Ql,self.Rl,Qrtens,Rrtens,rest1=cmf.regauge_with_trunc(self.Ql,[self.Rl],dx=0.0,gauge='symmetric',linitial=None,rinitial=None,nmaxit=100000,tol=1E-14,\
            #                                                                    ncv=40,numeig=6,pinv=1E-200,thresh=1E-10,trunc=1E-16,Dmax=self.Ql.shape[0],verbosity=0)
        #self.Rl=self.Rl[0]
    def doStep(self,cutoff=1.0,alpha=None,inter=0.0,invrange=1.0,operators=['n','n'],delta=1E-3,truncAfterFree=True,truncAfterInt=True,
               pinv=1E-200,tol=1E-12,Dthresh=1E-6,trunc=1E-10,Dinc=1,ncv=30,numeig=6,thresh=1E-8):
        """
        do a single evolution step

        Parameters:
        cutoff:    float (1.0)
                   UV cutoff of the cMERA
                   cutoff enters in the definition of the free entangler K0 (see below) 
                   as well as in the definition of the interacting entangler via the definition of 
                   psi(x) =\sqrt(cutoff/2)phi(x)+1/sqrt(2*cutoff)pi(x)
        alpha:     float or None
                   prefactor of the free entangler, if None, alpha=cutoff/4 is used
                   the free entangler has the form 
                   K0=alpha\int dx dy exp(-cutoff*abs(x-y)) :pi(x)phi(y):
                     =-1j*alpha/2\int dx dy exp(-cutoff*abs(x-y))*(psi(x)psi(y)-psidag(x)psidag(y))
                   to obtain the correct entangler for the ground-state of free boson with cutoff, use alpha=cutoff/4 (default)
        inter:     float
                   the interaction strength of the entangler
        invrange:  float
                   inverse length scale of the interaction
        operators: list of str
                   operators used to construct entangling propagator
                   case 1: for operators = ['n','n'], the entangler is given by inter * \int dx dy n(x) n(y) exp(-invrange*abs(x-y))
                           if len(operators)==2, operators=['n','n'] is the only allowed choice
                   case 2: for operators = [o1,o2,o3,o4], the entangler is given by inter * \int dw dx dy dx o1(w) o2(x) o3(y) o4(z) exp(-invrange abs(w-z))
                           if len(operators)==4, each element in operators can be one of the following :'phi','pi','psi','psidag'
        delta:    float
                  step-size;
                  propagators are constructed using delta
        truncAfterFree:  bool
                         if True, truncate the cMPS after application of the free propagator
                         Application order of propagators: 
                         1. free evolution
                         2. interacting evolution
                         3. rescaling
        truncAfterInt:   bool
                         if True, truncate the cMPS after application of the interacting propagator
                         for operators =['phi','phi','phi','phi'], bond dimension of the propagator 
                         is 4. Together with bond dimension 3 of free propagator, a full application 
                         without intermediate truncation increases D by a factor of 12, which can cause
                         slowness.
                         Application order of propagators: 
                         1. free evolution
                         2. interacting evolution
                         3. rescaling
        pinv:     float (1E-20)
                  pseudo-inverse parameter for inversion of the Schmidt-values and reduced density matrices
        tol:      float (1E-10):
                  precision parameter for calculating the reduced density matrices during truncation
        Dthresh:  float (1E-6)
                  threshold parameter; if the truncated weight of the last truncation is larger than Dthres,
                  the bond dimension D is increased by Dinc; if D is already at its maximally allowed value, 
                  D is not changed
        trunc:    float (1E-10)
                  truncation threshold during regauging; all Schmidt-values smaller than trunc will be removed, irrespective
                  of the maximally allowed bond-dimension
        Dinc:     int (1) 
                  bond-dimension increment
        ncv:      int (30)nn
                  number of krylov vectors to be used when calculating the transfer-matrix eigenvectors during truncation
        numeig:   int (6)
                  number of eigenvector-eigenvalue pairs of the transfer-matrix to be calculated 
        thresh:   float (1E-10)
                  related to printing some warnings; not relevant
        """
        
        self.cutoff=cutoff        
        if len(operators)==2:
            if not all([o=='n' for o in operators]):
                raise ValueError("unknown operators {}. If len(operators)==2, elements in operators can only be 'n'".format(np.array(operators)[[o != 'n' for o in operators]]))
            if abs(inter)>1E-10:            
                self.Gammaint=cmeralib.density_density_interactingEntanglingPropagator(invrange=invrange,delta=delta,inter=inter)
                self.Dmpoint=self.Gammaint[0][0].shape[0]
                interactiontype='nn'
        elif len(operators)==4:
            if not all([o in ('phi','pi','psi','psidag') for o in operators]):
                print()
                raise ValueError("unknown operators {}. If len(operators)==4, each element in operators has to be one of ('phi','pi','psi','psidag')".format(np.array(operators)[[o not in ('phi','pi','psi','psidag') for o in operators]]))
            if abs(inter)>1E-10:
                self.Gammaint=cmeralib.interactingEntanglingPropagator(cutoff=cutoff,invrange=invrange,delta=delta,inter=inter,operators=operators,dtype=self.dtype)
                self.Dmpoint=self.Gammaint[0][0].shape[0]
                interactiontype='oooo'
        else:
            raise ValueError("length of list 'operators' has to be 2 or 4")
        self.Gamma=cmeralib.freeEntanglingPropagator(cutoff=cutoff,delta=delta,alpha=alpha)
        self.Dmpo=self.Gamma[0][0].shape[0]
        
        if (self.truncated_weight>Dthresh) and (self.D<self.Dmax) and (len(self.lam)==self.D):
            self.D+=Dinc
            
        if (alpha==None) or (np.abs(alpha)>1E-10):
            self.Ql=np.kron(np.eye(self.Ql.shape[0]),self.Gamma[0][0])+np.kron(self.Ql,np.eye(self.Dmpo))+np.kron(self.Rl,self.Gamma[1][0])
            self.Rl=np.kron(np.eye(self.Rl.shape[0]),self.Gamma[0][1])+np.kron(self.Rl,np.eye(self.Dmpo))
            if truncAfterFree:
                try:                
                    self.lam,self.Ql,self.Rl,Qrtens,Rrtens,rest1=cmf.canonize(self.Ql,[self.Rl],linit=None,rinit=None,maxiter=100000,tol=tol,\
                                                                              ncv=ncv,numeig=numeig,pinv=pinv,thresh=thresh,trunc=trunc,Dmax=self.D,verbosity=0)
                    self.truncated_weight=np.sum(rest1)
                    self.Rl=self.Rl[0]
                except TypeError:
                    pass
                
        if np.abs(inter)>1E-10:
            self.Ql=np.kron(np.eye(self.Ql.shape[0]),self.Gammaint[0][0])+np.kron(self.Ql,np.eye(self.Dmpoint))
            if interactiontype=='nn':
                self.Rl=np.kron(self.Rl,self.Gammaint[1][1])
            elif interactiontype=='oooo':                
                self.Rl=np.kron(np.eye(self.Rl.shape[0]),self.Gammaint[0][1])+np.kron(self.Rl,np.eye(self.Dmpoint))
            if truncAfterInt:                
                try:
                    self.lam,self.Ql,self.Rl,Qrtens,Rrtens,rest2=cmf.canonize(self.Ql,[self.Rl],linit=None,rinit=None,maxiter=100000,tol=tol,\
                                                                              ncv=ncv,numeig=numeig,pinv=pinv,thresh=thresh,trunc=trunc,Dmax=self.D,verbosity=0)
                    self.truncated_weight+=np.sum(rest2)            
                    self.Rl=self.Rl[0]
                except TypeError:
                    pass
            
        self.Ql*=np.exp(-np.imag(delta))
        self.Rl*=np.exp(-np.imag(delta)/2.0)
        self.scale+=np.abs(delta)
        self.iteration+=1        
        
    def save(self,filename):
        """
        dump the cMERA instance into a pickle file "filename".pickle
        Parameters:
        ------------------
        filename: str()
                  the filename (without .pickle ending)
        """
        with open(filename+'.pickle','wb') as f:
            pickle.dump(self,f)

    @classmethod
    def load(cls,filename):
        """
        load a simulation from a pickle file
        Parameters:
        ---------------
        filename: str
                  the pickle file

        Returns:
        ---------------
        a cMERA instance holding the loaded simulation
        
        """
        with open(filename,'rb') as f:
            cls=pickle.load(f)
        return cls
    
    def addMonitoringVariables(self,data_accumulator):
        """
        adds some variables to data_accumulator. 
        data_accumulator: dict()
                          used to store variables of the cMERA simulation
                          the function maps the following keys to data and stores it in data_accumulator
                          'lams': the Schmidt-spectrum at the current step
                          'tw':   the last truncated weight
                          'D':    the current bond dimension
        """
        
        if 'lams' not in data_accumulator:
            data_accumulator['lams']=[self.lam]
        else:
            data_accumulator['lams'].append(self.lam)
        if 'tw' not in data_accumulator:
            data_accumulator['tw']=[self.truncated_weight]
        else:
            data_accumulator['tw'].append(self.truncated_weight)
        if 'D' not in data_accumulator:
            data_accumulator['D']=[len(self.lam)]
        else:
            data_accumulator['D'].append(len(self.lam))
        return data_accumulator

    
def calculateExactCorrelators(data_accumulator,scale,cutoff):
    """
    calculates the exact pi-pi correlation function at scale "scale", starting from an initial
    product state with UV-cutoff given by cutoff

    data_accumulator: dict():
                      the pi-pi correlation function is stored at key 'pipi_exact' as an np.ndarray
                      data_accumulator['pipi_exact']=correlation_function; 
    scale:            float
                      the scale at which to calculate the corelator
    cutoff:           float
                      UV-cutoff
    """
    xexact,pipiexact=cmeralib.FreeBosonPiPiCorrelatorExact(scale)
    if 'pipi_exact' not in data_accumulator:
        data_accumulator['pipi_exact']=[xexact/cutoff,pipiexact*cutoff**2]
    else:
        data_accumulator['pipi_exact'].append(pipiexact*cutoff**2)
        
    return data_accumulator


def calculatePiPiCorrelators(data_accumulator,cmera,N1=10,N2=40000,eps1=1E-4,eps2=4E-2):
    """
    calculates the pi-pi correlation function using the cMPS tensors from cmera
    and stores it in data_accumulator
    Parameters:
    ----------------------
    data_accumulator: dict():
                      the pi-pi correlation function is stored at key 'pipi' as an np.ndarray
                      data_accumulator['pipi']=correlation_function; 
    cmera:            cMERA instance
                      a cMERA simulation
    N1,eps1:          int,float
    N2,eps2:          int,float
                      the corralation function is calculated at points np.arange(N1)*eps1 and N1*eps1+np.arange(N2)*eps2
    Returns:
    ----------------------
    data_accumulator: dict()
                      see above
    """
    
    Qltens=cmera.Ql
    Rltens=cmera.Rl
    lamtens=cmera.lam

    x=np.append(np.arange(1,N1+1)*eps1,np.arange(2,N2+1)*eps2)
    pipi1,vec1=cmf.calculateRelativisticCorrelators(Ql=Qltens,Rl=Rltens,r=np.diag(lamtens**2),cutoff=cmera.cutoff,operators=['pi','pi'],dx=eps1,N=N1,initial=None)
    pipi2,vec2=cmf.calculateRelativisticCorrelators(Ql=Qltens,Rl=Rltens,r=np.diag(lamtens**2),cutoff=cmera.cutoff,operators=['pi','pi'],dx=eps2,N=N2,initial=vec1)
    #=cmf.PiPiCorr(Qltens,Rltens,np.diag(lamtens**2),eps1,N1,cmera.cutoff,initial=None)
    #=cmf.PiPiCorr(Qltens,Rltens,np.diag(lamtens**2),eps2,N2,cmera.cutoff,initial=vec1)
    pipi=np.append(pipi1,pipi2[1::])    

    if 'pipi' not in data_accumulator:
        data_accumulator['pipi']=[x,(cmera.scale,pipi)]
    else:
        data_accumulator['pipi'].append((cmera.scale,pipi))

    return data_accumulator

def calculatedPhidPhiCorrelators(data_accumulator,cmera,N1=10,N2=40000,eps1=1E-4,eps2=4E-2):
    """
    calculates the partial phi-partial phi correlation function using the cMPS tensors from cmera
    and stores it in data_accumulator
    Parameters:
    ----------------------
    data_accumulator: dict():
                      the partial phi-partial phi correlation function is stored at key 'dphidphi' as an np.ndarray
                      data_accumulator['pipi']=correlation_function; 
    cmera:            cMERA instance
                      a cMERA simulation
    N1,eps1:          int,float
    N2,eps2:          int,float
                      the corralation function is calculated at points np.arange(N1)*eps1 and N1*eps1+np.arange(N2)*eps2
    Returns:
    ----------------------
    data_accumulator: dict()
                      see above

    """
    
    Qltens=cmera.Ql
    Rltens=cmera.Rl
    lamtens=cmera.lam

    x=np.append(np.arange(1,N1+1)*eps1,np.arange(2,N2+1)*eps2)
    dxphidxphi1,vec1=cmf.calculateRelativisticCorrelators(Ql=Qltens,Rl=Rltens,r=np.diag(lamtens**2),cutoff=cmera.cutoff,operators=['p_phi','p_phi'],dx=eps1,N=N1,initial=None)
    dxphidxphi2,vec2=cmf.calculateRelativisticCorrelators(Ql=Qltens,Rl=Rltens,r=np.diag(lamtens**2),cutoff=cmera.cutoff,operators=['p_phi','p_phi'],dx=eps2,N=N2,initial=vec1)
    #dxphidxphi1,vec1=cmf.dxPhidxPhiCorr(Qltens,Rltens,np.diag(lamtens**2),eps1,N1,cmera.cutoff,initial=None)
    #dxphidxphi2,vec2=cmf.dxPhidxPhiCorr(Qltens,Rltens,np.diag(lamtens**2),eps2,N2,cmera.cutoff,initial=vec1)
    dxphidxphi=np.append(dxphidxphi1,dxphidxphi2[1::])

    if 'dphidphi' not in data_accumulator:
        data_accumulator['dphidphi']=[x,(cmera.scale,dxphidxphi)]
    else:
        data_accumulator['dphidphi'].append((cmera.scale,dxphidxphi)) 
    return data_accumulator


def calculatePsiObservables(data_accumulator,cmera):
    """
    calculates the observable <psi> using the cMPS tensors from cmera 
    and stores it in data_accumulator
    Parameters:
    ----------------------
    data_accumulator: dict():
                      the partial phi-partial phi correlation function is stored at key 'psi' as an np.ndarray
                      data_accumulator['pipi']=correlation_function; 
    cmera:            cMERA instance
                      a cMERA simulation
    Returns:
    ----------------------
    data_accumulator: dict()
                      see above

    """
    
    Qltens=cmera.Ql
    Rltens=cmera.Rl
    lamtens=cmera.lam
    
    psi=np.trace(Rltens.dot(np.diag(lamtens)).dot(np.diag(lamtens)))
    
    if 'psi' not in data_accumulator:
        data_accumulator['psi']=[psi]
    else:
        data_accumulator['psi'].append(psi)        
    return data_accumulator

def calculateDensityObservables(data_accumulator,cmera):
    """
    calculates the observable <psidag psi> (particle density) using the cMPS tensors from cmera
    and stores it in data_accumulator
    Parameters:
    ----------------------
    data_accumulator: dict():
                      the partial phi-partial phi correlation function is stored at key 'density' as an np.ndarray
                      data_accumulator['pipi']=correlation_function; 
    cmera:            cMERA instance
                      a cMERA simulation
    Returns:
    ----------------------
    data_accumulator: dict()
                      see above

    """
    
    Qltens=cmera.Ql
    Rltens=cmera.Rl
    lamtens=cmera.lam
    
    dens=np.trace(Rltens.dot(np.diag(lamtens)).dot(np.diag(lamtens)).dot(herm(Rltens)))
    
    if 'density' not in data_accumulator:
        data_accumulator['density']=[dens]
    else:
        data_accumulator['density'].append(dens)        
        data_accumulator['psi'].append(psi)        
    return data_accumulator

def checkWicksTheorem(data_accumulator,cmera,N=20000,eps=0.01):
    Qltens=cmera.Ql
    Rltens=cmera.Rl
    lamtens=cmera.lam
    
    x=np.arange(1,N+1)*eps
    wick=cmeralib.checkWickTheorem(Qltens,Rltens,np.diag(lamtens**2),eps,N)
    if 'wick_theorem' not in data_accumulator:
        data_accumulator['wick_theorem']=[x,wick]
    else:
        data_accumulator['wick_theorem'].append(wick)
    return data_accumulator

def convert(val):

    """
    converts an input str() val into its numeric type
    see the conversion rules below
    
    """
    if val=='True':
        return True
    elif val=='False':
        return False
    elif val=='None':
        return None
    else:
        types=[int,float,str]
    for t in types:
        try:
            return t(val)
        except ValueError:
            pass
        
def read_parameters(filename):
    """
    read parameters from a file "filename"
    the file format is assumed to be 

    parameter1 value1
    parameter2 value2
        .
        .
        .

    or 

    parameter1: value1
    parameter2: value2

    Returns:
    python dict() containing mapping parameter-name to its value
    """
    
    params={}
    with open(filename, 'r') as f:
        for line in f:
            if '[' not in line:
                params[line.replace(':','').split()[0]]=convert(line.replace(':','').split()[1])
            else:
                s=re.sub('[\[\]:,\']','',line).split()
                params[s[0]]=[convert(t) for t in s[1::]]
    return params



def plot(data_accumulator,title='',which=('pipi','dphidphi','lam','density','psi','tw','wick')):
    """
    plots data from data_accumulator
    Parameters:
    ---------------------------------
    data_accumulator:  dict()
                       holds the data to be plotted. The data has to be obtained from one of the data-generating
                       functions above
    title:             str()
                       an optional title for the plots
    which:             tuple() of str()
                       the names of the data to be plotted. Each element in which has to be a key
                       in data_accumulator. 
    """
    plt.ion()
    if ('pipi' in which) and ('exact' not in which):
        try:
        
            plt.figure(1)
            plt.clf()
            plt.subplot(2,1,1)
            plt.title(title)
            plt.loglog(data_accumulator['pipi'][0],np.abs(data_accumulator['pipi'][-1][1]),'-b')
            plt.ylim([1E-12,100])
            plt.ylabel(r'$|\langle\pi(x)\pi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.tight_layout()
            
            plt.subplot(2,1,2)
            plt.title(title)
            plt.semilogx(data_accumulator['pipi'][0],np.abs(data_accumulator['pipi'][-1][1]),'-b')
            plt.ylabel(r'$|\langle\pi(x)\pi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.legend(['numerical evolution'],loc='best')    
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)
            
        except KeyError:
            pass
        
    if ('pipi' not in which) and ('exact' in which):
        try:
            plt.figure(1)
            plt.clf()
            plt.subplot(2,1,1)
            plt.title(title)
            plt.loglog(data_accumulator['pipi_exact'][0],data_accumulator['pipi_exact'][-1],'--k')
            plt.ylim([1E-12,100])
            plt.ylabel(r'$|\langle\pi(x)\pi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.tight_layout()
            
            plt.subplot(2,1,2)
            plt.title(title)
            plt.semilogx(data_accumulator['pipi_exact'][0],data_accumulator['pipi_exact'][-1],'--k')
            plt.ylabel(r'$|\langle\pi(x)\pi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.legend(['exact free'],loc='best')    
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)

        except KeyError:
            pass
            
    if ('pipi' in which) and ('exact'in which):
        try:
            plt.figure(1)
            plt.clf()
            plt.subplot(2,1,1)
            plt.title(title)
            plt.loglog(data_accumulator['pipi'][0],np.abs(data_accumulator['pipi'][-1][1]),'-b',data_accumulator['pipi_exact'][0],data_accumulator['pipi_exact'][-1],'--k')
            plt.ylim([1E-12,100])
            plt.ylabel(r'$|\langle\pi(x)\pi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.tight_layout()
            
            plt.subplot(2,1,2)
            plt.title(title)
            plt.semilogx(data_accumulator['pipi'][0],np.abs(data_accumulator['pipi'][-1][1]),'-b',data_accumulator['pipi_exact'][0],data_accumulator['pipi_exact'][-1],'--k')
            plt.ylabel(r'$|\langle\pi(x)\pi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.legend(['numerical evolution','exact free'],loc='best')    
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)

        except KeyError:
            pass
        
    if 'dphidphi' in which:
        try:
            plt.figure(2)
            plt.clf()
            plt.subplot(2,1,1)
            plt.title(title)    
            plt.loglog(data_accumulator['dphidphi'][0],np.abs(data_accumulator['dphidphi'][-1][1]),'-b')
            plt.ylabel(r'$|\langle\partial\phi(x)\partial\phi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.ylim([1E-12,100])
            plt.tight_layout()
            
            plt.subplot(2,1,2)
            plt.title(title)
            plt.semilogx(data_accumulator['dphidphi'][0],np.abs(data_accumulator['dphidphi'][-1][1]),'-b')    
            plt.ylabel(r'$|\langle\partial\phi(x)\partial\phi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.legend(['numerical evolution','exact free'],loc='best')
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)
            
        except KeyError:
            pass
        
    if 'lam' in which:
        try:
            plt.figure(3)
            plt.title(title)
            plt.semilogy(data_accumulator['scale'][-1],[data_accumulator['lams'][-1]],'bx')        
            plt.ylabel(r'$\lambda_i$',fontsize=20)
            plt.xlabel(r'$scale$',fontsize=20)
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)
        except KeyError:
            pass

    if 'density' in which:
        try:
            plt.figure(4)
            plt.clf()
            plt.title(title)
            plt.plot(data_accumulator['scale'],data_accumulator['density'],'bd',markersize=6)
            plt.ylabel(r'$\langle n\rangle$',fontsize=20)
            plt.xlabel(r'$scale$',fontsize=20)
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)
        except KeyError:
            pass
    
    if 'psi' in which:
        try:
            plt.figure(5)
            plt.clf()
            plt.title(title)
            plt.plot(data_accumulator['scale'],data_accumulator['psi'],'bd',markersize=6)            
            plt.ylabel(r'$\langle \psi\rangle$',fontsize=20)
            plt.xlabel(r'$scale$',fontsize=20)
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)
        except KeyError:
            pass
        
    if 'tw' in which:
        try:
            plt.figure(6)
            plt.clf()
            plt.title(title)
            plt.plot(data_accumulator['scale'],data_accumulator['tw'],'bd',markersize=6)            
            plt.ylabel(r'tw',fontsize=20)
            plt.xlabel(r'$scale$',fontsize=20)
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)
        except KeyError:
            pass
        
    if 'wick' in which:
        try:
            plt.figure(7)
            plt.clf()
            plt.title(title)
            plt.semilogy(data_accumulator['wick_theorem'][0],np.abs(data_accumulator['wick_theorem'][-1]))
            plt.ylabel(r'violation of wicks th.',fontsize=20)
            plt.xlabel(r'$x$',fontsize=20)
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)
        except KeyError:
            pass
    plt.ioff()


    
if __name__ == "__main__":

    parser = argparse.ArgumentParser('cMERA.py')
    parser.add_argument('--info_cMERA', help='print a small manual',action='store_true')    
    parser.add_argument('--delta',help='imaginary part of the time step for unitary time evolution (0.001)',type=float,default=0.001)
    parser.add_argument('--Dmax', help='maximal cMPS bond dimension (32)',type=int,default=32)
    parser.add_argument('--nwarmup', help='number of initial warmup-steps without truncation (2); ',type=int,default=2)    
    parser.add_argument('--Dinc', help='bond dimension increment (1)',type=int,default=1)    
    parser.add_argument('--cutoff', help='UV cutoff of the entangler (1.0)',type=float,default=1.0)
    parser.add_argument('--alpha', help='entangling strength; if not given, alpha=cutoff/4 (None)',type=float,default=None)    
    parser.add_argument('--invrange', help='inverse interctionrange (1.0)',type=float,default=1.0)    
    parser.add_argument('--pinv',help='pseudoinver cutoff (1E-20); if chosen too large, severe artifacts will show up',type=float,default=1E-20)
    parser.add_argument('--operators', nargs='+',help="list of length 2 or 4 of str. \n for length 2: elements have to be  'n'; \n for length 4: use any of the following: ['pi','phi','psi','psidag']",type=str,default=['n','n'])
    parser.add_argument('--trunc',help='truncation threshold (1E-10); all schmidt-values below trunc will be discarded, irrespective of Dmax',type=float,default=1E-10)
    parser.add_argument('--Dthresh',help='truncation threshold at which the bond dimension is increased by Dinc (1E-6)',type=float,default=1E-6)    
    parser.add_argument('--thresh',help='threshold for "large-imaginary-eigenvalue" error (1E-10); dont worry about it',type=float,default=1E-10)        
    parser.add_argument('--tol', help='tolerance of eigensolver for finding left and right reduced DM (1E-10)',type=float,default=1E-10)
    parser.add_argument('--imax', help='maximum number of iterations (5000)',type=int,default=5000)
    parser.add_argument('--checkpoint', help='save the simulation every checkpoint iterations for checkpointing (100)',type=int,default=100)
    parser.add_argument('--resume_checkpoint', help='load a checkpointed file and resume simulation',type=str)
    parser.add_argument('--filename', help='filename for output (_interactingBosoncMERA)',type=str,default='_interactingBosoncMERA')
    parser.add_argument('--truncAfterFree', help='apply truncation after free propagation (True)',action='store_true')
    parser.add_argument('--truncAfterInt', help='apply truncation after interacting propagation (True)',action='store_true')
    parser.add_argument('--loaddir', help='filename of the simulation to be loaded; the resumed simulation will be stored in filename (see above)',type=str)
    parser.add_argument('--parameterfile', help='read parameters from a given file; each line in the file has to contain the parameter name and its value seperated by a whitespace; values passed by file override values passed by command line',type=str)    
    parser.add_argument('--ending', help='suffix of the file names: Ql+args.ending, Rl+args.ending, lam+args.ending ',type=str)
    parser.add_argument('--numeig', help='number of eigenvector in TMeigs (5)',type=int,default=5)
    parser.add_argument('--ncv', help='number of krylov vectors in TMeigs (40)',type=int,default=40)
    parser.add_argument('--show_plots', nargs='+',help='list of strings from {pipi,exact,dphidphi,density,psi,lams,tw}',type=str,default=[''])
    parser.add_argument('--measurestep', help='calculate observables ever measurestep; if 0, nothing is measured (0)',type=int,default=0)
    parser.add_argument('--measure', nargs='+',help='list of strings from {pipi,exact,dphidphi,density,psi,lams,tw}',type=str,default=[''])    
    parser.add_argument('--inter', help='interaction (0.0)',type=float,default=0.0)
    parser.add_argument('--keepcp', help='keep old checkpoint files of the simulation',action='store_true')
    parser.add_argument('--N1', help='number of points for calculating correlators at distances np.arange(N1**eps1 (10)',type=int,default=10)
    parser.add_argument('--N2', help='number of points for calculating correlators at distances eps1*N1+np.arange(N2)*eps2 (40000)',type=int,default=40000)
    parser.add_argument('--N3', help='number of points for calculating violation of wicks theorem at distances eps3*N3 (20000)',type=int,default=20000)        
    parser.add_argument('--eps1', help='discretization for calculating correlators',type=float,default=1E-4)
    parser.add_argument('--eps2', help='discretization for calculating correlators',type=float,default=1E-2)
    parser.add_argument('--eps3', help='discretization for calculating violation of wicks theorem',type=float,default=1E-2)        

    args=parser.parse_args()
    if args.info_cMERA:
        help(cMERA)
        sys.exit()
    observables=['pipi','dphidphi','lam','density','psi','tw','wick','exact']
    if args.parameterfile!=None:
        parameters=read_parameters(args.parameterfile)
        for k,v in parameters.items():
            if (k!='parameterfile') and (k!='filename'):#read all parameters except filename and parameterfile
                setattr(args,k,v)

    if not all([s in observables for s in args.show_plots]):                
        warnings.warn('cannot plot unknown quantity/quantities {0}'.format(np.asarray(args.show_plots)[[s not in observables for s in args.show_plots]]),stacklevel=2)

    if not all([s in observables for s in args.measure]):                
        warnings.warn('cannot measure unknown quantity/quantities {0}'.format(np.asarray(args.measure)[[s not in observables for s in args.measure]]),stacklevel=2)
        
    if not all([s in args.measure for s in args.show_plots]):                
        warnings.warn('observables {0} are not measured, but are passed to --show_plots; skipping plots'.format(np.asarray(args.show_plots)[[s not in args.measure for s in args.show_plots]]),stacklevel=2)


    if args.show_plots!=[''] and not any([s in args.measure for s in observables]):                
        warnings.warn('some parameters have been passed to --show_plots, but no measurements are specified in --measure; skipping plots ',stacklevel=2)
        
    if (args.measurestep!=0):
        if not any([s in args.measure for s in observables]):
            warnings.warn('a nonzero measurestep --measurestep has been given, but no measurement has been specified in --measure',stacklevel=2)


    date=datetime.datetime.now()
    today=str(date.year)+str(date.month)+str(date.day)
    folder=today+args.filename+'delta_{0}_Dmax{1}_cutoff{2}_inter{3}_invrange{4}'.format(args.delta,args.Dmax,args.cutoff,args.inter,args.invrange)
    filename=args.filename+'delta_{0}_Dmax{1}_cutoff{2}_inter{3}_invrange{4}'.format(args.delta,args.Dmax,args.cutoff,args.inter,args.invrange)
    
    if os.path.exists(folder):
        N=20
        print ('Found folder identical to simulation name {0}. Resuming simulation will overwrite existing data.\nYou have {1} seconds to hit CTRL-C to abort.\n Enter any number to continue now'.format(filename,N))
        for n in range(N,0,-1):
            sys.stdout.write("\rResuming in %i seconds" % n)
            sys.stdout.flush()
            a,b,c=select.select([sys.stdin],[],[],1)
            if sys.stdin in a:
                #catches an enter and resumes immediatly
                input()    
                break
            
    if not os.path.exists(folder):
        os.mkdir(folder)
        
    root=os.getcwd()
    os.chdir(folder)
                
    parameters=vars(args)
    with open('parameters.dat','w') as f:
        for k,v in parameters.items():
            f.write(k + ' {0}\n'.format(v))
    f.close()
        
    #store all initialization parameters in a dict()
    init_params=dict(cutoff=args.cutoff,
                     alpha=args.alpha,
                     inter=args.inter,
                     invrange=args.invrange,
                     operators=args.operators,
                     delta=args.delta*1.0j,
                     nwarmup=args.nwarmup,
                     Dmax=args.Dmax,
                     dtype=complex)
    
    #store all evolution parameters in a dict()
    if (not args.truncAfterInt) and (not args.truncAfterFree):
        setattr(args,'truncAfterFree',True)
    evolution_params=dict(cutoff=args.cutoff,
                          alpha=args.alpha,
                          inter=args.inter,
                          invrange=args.invrange,
                          operators=args.operators,
                          delta=args.delta*1.0j,
                          truncAfterFree=args.truncAfterFree,
                          truncAfterInt=args.truncAfterInt,                          
                          pinv=args.pinv,
                          tol=args.tol,
                          Dthresh=args.Dthresh,
                          trunc=args.trunc,
                          Dinc=args.Dinc,
                          ncv=args.ncv,
                          numeig=args.numeig,
                          thresh=args.thresh)
    
    cmera_sim=cMERA(**init_params)
    data_accumulator=dict(scale=[])
    last_stored=None
    for step in range(args.imax):
        if args.checkpoint!=0 and cmera_sim.iteration>0 and step%args.checkpoint==0:
            if args.keepcp==True:
                last_stored=f'cmera_checkpoint_{cmera_sim.iteration}'
                cmera_sim.save(last_stored)                
            else:
                if (last_stored!=None):# and os.path.exists(last_stored):
                    try:
                        os.remove(last_stored+'.pickle')
                    except OSError:
                        pass
                last_stored=f'cmera_checkpoint_{cmera_sim.iteration}'
                cmera_sim.save(last_stored)                

        cmera_sim.doStep(**evolution_params)
        if (args.measurestep!=0) and (cmera_sim.iteration%args.measurestep==0):
            if 'pipi' in args.measure:
                data_accumulator=calculatePiPiCorrelators(data_accumulator,cmera_sim)
            if 'dphidphi' in args.measure:
                data_accumulator=calculatedPhidPhiCorrelators(data_accumulator,cmera_sim)
            if ('exact' in args.measure):
                data_accumulator=calculateExactCorrelators(data_accumulator,cmera_sim.scale,args.cutoff)
            if 'psi' in args.measure:
                data_accumulator=calculatePsi(data_accumulator,cmera_sim)
            if 'density' in args.measure:
                data_accumulator=calculateDensity(data_accumulator,cmera_sim)
            if ('lams' in args.measure) or ('tw' in args.measure):                                
                data_accumulator=cmera_sim.addMonitoringVariables(data_accumulator)
            if 'wick' in args.measure:
                data_accumulator=checkWicksTheorem(data_accumulator,cmera_sim,N=args.N3,eps=args.eps3)
            if any([s in args.measure for s in observables]):                
                data_accumulator['scale'].append(cmera_sim.scale)
                with open('data_accumulator'+'.pickle','wb') as f:
                    pickle.dump(data_accumulator,f)

            if args.show_plots!=['']:
                if any([s in args.measure for s in observables]) and any(s in args.measure for s in args.show_plots):                
                    plot(data_accumulator,title='',which=tuple([s for s in args.show_plots if s in args.measure]))#plot only those that are measured
                else:
                    warnings.warn('nothing to plot; skipping plots',stacklevel=2)
                
        sys.stdout.write('\r iteration %i, D=%i Dmax=%i, tw %.10f' %(cmera_sim.iteration,len(cmera_sim.lam),cmera_sim.Dmax,cmera_sim.truncated_weight))
        sys.stdout.flush()

